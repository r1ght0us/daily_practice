import sys
from music21 import *
import numpy as np
from models.grammar import *
from models.qa import *
from models.preprocess import *
from models.music_utils import *
from models.data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error

# 加载数据集
X, Y, n_values, indices_values = load_music_utils()

# 建立模型
n_a = 64  # 设置64个LSTM
reshapor = Reshape((1, 90))  # 重塑为(1,90)张量
LSTM_cell = LSTM(n_a, return_state=True)  # 输出空间维度为64，除了输出之外返回最后一个状态
densor = Dense(n_values, activation="softmax")  # 全连接层，输出空间维度是90,全连接层，激活函数为softmax


def djmodel(Tx, n_a, n_values):
    """
    Tx: 语料库长裤
    n_a: LSTMcell数量
    n_values: 音乐数据中唯一值的数量
    """
    X = Input(shape=(Tx, n_values))  # 输入的形状为(Tx,n_values)
    a0 = Input(shape=(n_a,), name="a0")  # 初始a0
    c0 = Input(shape=(n_a,), name="c0")  # 每个输入都应有c0的维度表示
    a = a0
    c = c0

    outputs = []  # 定义空列表收集迭代信息

    for t in range(Tx):
        x = Lambda(lambda x: X[:, t, :])(X)  # 将任意表达式封装成Layer对象，选择第t个时间步向量
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])  # initial_state 传递给单元格的第一个调用的初始状态张量的列表
        out = densor(a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)  # 创建模型实体

    return model


model = djmodel(Tx=30, n_a=64, n_values=90)
# model.summary()  # 打印出模型概述信息

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)  # decay每次参数更新后学习率衰减值
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])  # 配置训练模型

m = 60
a0 = np.zeros((m, n_a))  # 初始化
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=100)  # 训练模型


def music_inference_model(LSTM_cell, densor, n_values=90, n_a=64, Ty=100):
    """
    LSTM_cell -- 来自model()的训练过后的LSTM单元，是keras层对象。
    densor -- 来自model()的训练过后的"densor"，是keras层对象
    Ty -- 整数，生成的是时间步的数量
    """
    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0  # cell的状态类似于y^<t>输出
    x = x0

    outputs = []

    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        out = densor(a)  # 使用densor()应用于LSTM_Cell的隐藏状态输出
        outputs.append(out)

        x = Lambda(one_hot)(out)  # 根据“out”选择下一个值，并将“x”设置为所选值的一个独热编码

    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


inference_model = music_inference_model(LSTM_cell, densor, n_values=90, n_a=64, Ty=50)  # 经过硬编码以生成50个值

x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    使用模型预测当前值的下一个值。
    """
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])  # 为输入样本生成输出预测
    indices = np.argmax(pred, axis=-1)  # 沿轴返回最大值的索引，axis=-1等同于python列表对于-1的定义相同
    results = to_categorical(indices, num_classes=90)  # 将类向量（整数）转换为二进制类矩阵。

    return results, indices


results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

out_stream = generate_music(inference_model)
