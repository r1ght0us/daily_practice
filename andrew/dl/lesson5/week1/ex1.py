import numpy as np
from models.rnn_utils import *


# RNN单元

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Wax, xt)+np.dot(Waa, a_prev)+ba)  # 计算a^<t>

    yt_pred = softmax(np.dot(Wya, a_next)+by)  # 计算y_hat

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

# RNN 正向传播


def rnn_forward(x, a0, parameters):
    caches = []

    n_x, m, T_x = x.shape  # 输入整个X的序列
    n_y, n_a = parameters["Wya"].shape

    a = np.zeros((n_a, m, T_x))  # 该向量将存储RNN计算的所有隐藏状态
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(
            x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next  # 保存中间的隐藏值
        y_pred[:, :, t] = yt_pred

        caches.append(cache)

    caches = (caches, x)  # 存储反向传播所用的东西

    return a, y_pred, caches


# LSTM单元
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters["Wf"]  # 遗忘门参数
    bf = parameters["bf"]
    Wi = parameters["Wi"]  # 更新门参数
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]   # 进行tanh所需要的参数
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.zeros((n_x+n_a, m))  # 将a^(t-1)和x^<t>以列链接的方式在一起
    concat[:n_a, :] = a_prev
    concat[n_a:, :] = xt

    ft = sigmoid(np.dot(Wf, concat)+bf)  # 遗忘门
    it = sigmoid(np.dot(Wi, concat)+bi)  # 更新门
    cct = np.tanh(np.dot(Wc, concat)+bc)  # 进行tanh
    c_next = ft*c_prev+it*cct  # 输出c^<t>
    ot = sigmoid(np.dot(Wo, concat)+bo)  # 输出门

    a_next = ot*np.tanh(c_next)  # 输出a^<t>

    yt_pred = softmax(np.dot(Wy, a_next)+by)  # 输出y^<t>

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


# LSTM正向传播

def lstm_forward(x, a0, parameters):
    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(
            x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y[:, :, t] = yt
        c[:, :, t] = c_next
        caches.append(cache)
    
    caches=(caches,x)

    return a,y,c,caches

    