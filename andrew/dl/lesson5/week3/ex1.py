# 建立一个神经机器翻译（NMT）模型，以将人类可读的日期（"25th of June, 2009"）转换为机器可读的日期（"2009-06-25"）。

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model

import keras.backend as K
import numpy as np

from faker import Faker
import random
from tensorflow.python.eager.context import context
from tqdm import tqdm_notebook as tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
'''
dataset ：（人可读日期，机器可读日期）元组列表
human_vocab：python字典，将人类可读日期中使用的所有字符映射到整数索引
machine_vocab：python字典，将机器可读日期中使用的所有字符映射到整数索引。这些索引不一定与human_vocab一致。
inv_machine_vocab：machine_vocab的逆字典，从索引映射回字符。
'''

Tx = 30  # 输入长度
Ty = 10  # 输出长度
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
'''
X：训练集中人类可读日期的处理版本，其中每个字符都由通过human_vocab映射到该字符的索引替换。每个日期都用特殊字符（<pad>）进一步填充为Tx值。X.shape = (m, Tx)
Y：训练集中机器可读日期的处理版本，其中每个字符都被映射为machine_vocab中映射到的索引替换。你应该具有Y.shape = (m, Ty)。
Xoh：X的一个独热版本，由于human_vocab，将“1”条目的索引映射到该字符。Xoh.shape = (m, Tx, len(human_vocab))
Yoh：Y的一个独热版本，由于使用machine_vocab，因此将“1”条目的索引映射到了该字符。Yoh.shape = (m, Tx, len(machine_vocab))在这里，因为有11个字符（“-”以及0-9），所以len(machine_vocab) = 11。

'''


# 如果你必须将一本书的段落从法语翻译为英语，则无需阅读整个段落然后关闭该书并进行翻译。即使在翻译过程中，你也会阅读/重新阅读并专注于与你所写下的英语部分相对应的法语段落部分。
# 带注意力机制的神经机器翻译

repeator = RepeatVector(Tx)  # 将输入重复n次
concatenator = Concatenate(axis=-1)  # 所有输入张量通过 axis 轴串联起来的输出张量
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, activation="relu")
activator = Activation(softmax, name="attention_weights")
dotor = Dot(axes=1)  # d对应轴进行点乘


def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)  # 复制Tx个s_prev
    concat = concatenator([a, s_prev])  # 将其联合
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)  # softmax获得注意力权重
    context = dotor([alphas, a])  # 最后获得上下文

    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name="s0")  # n_s是post-attention LSTM的大小
    c0 = Input(shape=(n_s,),)
    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(
        X)  # RNN 的双向封装器，对序列进行前向和后向计算。

    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)  # 应用softmax
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

#model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

model.load_weights('models/model.h5')

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007',
            'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    s0=np.zeros((1,n_s))
    c0=np.zeros((1,n_s))
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(
        list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))
