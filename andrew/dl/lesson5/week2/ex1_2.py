# 建立一个LSTM模型作为输入单词序列

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from models.emo_utils import *


X_train, Y_train = read_csv("data/train_emoji.csv")
X_test, Y_test = read_csv("data/tesss.csv")
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    "data/glove.6B.50d.txt")
maxLen = len(max(X_train, key=len).split())

# 将X（字符串形式的句子数组）转换为与句子中单词相对应的索引数组


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    # 画了个二维矩阵，然后每一个i代表一个句子，j是每个句子的每个单词，并且转换为索引
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = (X[i].lower()).split()

        j = 0

        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index)+1  # embedding要求input_dim+1
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # 词向量维度，输入序列长度

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():  # 返回可遍历的(键, 值) 元组数组。
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(
        vocab_len, emb_dim, trainable=False)  # 设置该层不可训练
    embedding_layer.build((None,))  # 创建该层的权重为none
    embedding_layer.set_weights([emb_matrix])  # 设置该层的权重

    return embedding_layer


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype="int32")
    embedding_layer = pretrained_embedding_layer(
        word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)  # 返回一个词嵌入

    X = LSTM(128, return_sequences=True)(embeddings)  # 返回全部序列
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)  # 返回输出序列的最后一个输出
    X = Dropout(0.5)(X)
    X = Dense(5)(X)  # 使网络中过度拟合的影响最小化
    X = Activation("softmax")(X)

    model = Model(sentence_indices, X)

    return model


model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# 将X_train（作为字符串的句子数组）转换为X_train_indices（作为单词索引列表的句子数组）
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=5)

model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

# 在测试集上运行模型
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C=5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

# 查看错误示例
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:' + label_to_emoji(
            Y_test[i]) + ' prediction: ' + X_test[i] + label_to_emoji(num).strip())
