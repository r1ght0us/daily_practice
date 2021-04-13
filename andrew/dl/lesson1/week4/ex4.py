# 构建深度学习网络
import numpy as np
import numpy.random
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))

# 初始化
# 2层神经网络


def initialize_parameters(n_x, n_h, n_y):
    '''
    模型的结构为：LINEAR -> RELU -> LINEAR -> SIGMOID。
    随机初始化权重矩阵。 确保准确的维度，使用np.random.randn（shape）* 0.01。
    将偏差初始化为0。 使用np.zeros（shape）。
    '''
    W1 = rs.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = rs.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

# L层神经网络
# L层神经网络初始化


def initialize_parameters_deep(layer_dims):
    '''
    模型结构：[LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID(L-1层使用ReLU作为激活函数，最后一层采用sigmoid激活函数输出。)
    layer_dims存储存储n^[l]，即不同的层神经元数，例如：layer_dims为[2,4,1]：即有两个输入，一个隐藏层包含4个隐藏单元，一个输出层包含1个输出单元。
    '''
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        #parameters["W"+str(l)] = rs.randn(layer_dims[l], layer_dims[l-1])*0.01 #产生梯度消失
        parameters["W"+str(l)] = rs.randn(layer_dims[l], layer_dims[l-1])/ np.sqrt(layer_dims[l - 1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# 正向传播模块


def linear_forward(A, W, b):
    Z = np.dot(W, A)+b
    cache = (A, W, b)
    return Z, cache

# 正向线性激活


def linear_activation_forward(A_prev, W, b, activation):
    '''
    两个激活函数：S函数和Relu函数
    '''
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    # linear_cachae返回(A,W,b)、activation_cache返回Z
    cache = (linear_cache, activation_cache)
    return A, cache

# L层模型


def L_model_forward(X, parameters):
    '''
    parameters是initialize_parameters_deep函数的返回值，必为整数且能被2整除
    X:输入变量
    '''
    caches = []
    A = X
    L = len(parameters)//2  # 返回整数
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, parameters["W"+str(L)], parameters["b"+str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

# 损失函数


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL), axis=1, keepdims=True)
    cost = np.squeeze(cost)
    return cost

# 反向传播模块

# 线性反向


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

# 反向线性激活


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    else:
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}

    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL)-np.divide(1-Y, 1-AL))  # 相对于AL的导数

    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)
                                                    ] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
           grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp

    return grads

# 更新参数


def update_parameters(parameters, grads, learing_rate):
    L = len(parameters)//2

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - \
            learing_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - \
            learing_rate*grads["db"+str(l+1)]
    return parameters

