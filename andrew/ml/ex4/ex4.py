# ex4data1.mat包含5000张20*20像素的手写数字图片，读取需要使用SciPy工具
# ex4weights.mat为一组神经网络权重，第一层为输入层，输入为20*20像素手写图片的400个像素值；第二层为隐藏层，有25个单元；第三层为输出层，10个单元，分别对应10个数字的判断

# 神经网络
# 数据可视化

from sklearn.metrics import classification_report  # 这个包是评价报告
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

data = loadmat("ex4data1.mat")
x = data["X"]
y = data["y"]
# print(x.shape,y.shape)

weight = loadmat("ex4weights.mat")
theta1, theta2 = weight["Theta1"], weight["Theta2"]
# print(theta1.shape,theta2.shape)

sample_idx = np.random.choice(
    np.arange(data["X"].shape[0]), 100)  # 随机选取100个在此区间的数字
sample_images = data["X"][sample_idx, :]  # 从data["X"]矩阵中随机选取100行建立新矩阵
fig, ax_array = plt.subplots(
    nrows=10, ncols=10, sharex=True, sharey=True, figsize=(12, 8))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(
            sample_images[10*r+c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
        plt.xticks([])  # 传递空列表将会删除x轴标签
        plt.yticks([])
# plt.show()

# 模型展示
# 前向传播和代价函数


def S(z):
    return 1/(1+np.exp(-z))

# 前向传播函数(见图前向传播原理)


def forward_propgate(x, theta1, theta2):
    m = x.shape[0]
    a1 = np.insert(x, 0, values=np.ones(m), axis=1)  # 对X插入一列(axis=1)
    z2 = a1*theta1.T
    a2 = np.insert(S(z2), 0, values=np.ones(m), axis=1)
    z3 = a2*theta2.T
    h = S(z3)
    return a1, z2, a2, z3, h

# 代价函数


def cost(theta1, theta2, x, y):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    a1, z2, a2, z3, h = forward_propgate(x, theta1, theta2)

    # 计算代价函数
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1-y[i, :]), np.log(1-h[i, :]))
        J = J+np.sum(first_term-second_term)

    J = J/m
    return J


# 对y标签进行编码

encoder = OneHotEncoder(sparse=False)  # 返回数组
y_onehot = encoder.fit_transform(y)  # 对y进行编码转换
# print(y_onehot.shape)
# print(cost(theta1,theta2,x,y_onehot))

# 正则化代价函数


def costREG(theta1, theta2, x, y, learning_rate):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    a1, z2, a2, z3, h = forward_propgate(x, theta1, theta2)

    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)  # 将数组中的元素相加
    J = J/m
    J = J+(float(learning_rate)/(2*m)) * \
        (np.sum(np.power(theta1[:, 1:], 2)) +
         np.sum(np.power(theta2[:, 1:], 2)))  # 正则化
    return J

# 反向传播
# S函数梯度


def S_gradient(z):
    return np.multiply(S(z), (1-S(z)))


# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
# 随机初始化

params = (np.random.random(size=hidden_size*(input_size+1) +
                           num_labels*(hidden_size+1))-0.5)*0.24

# 反向传播


def backrop(params, input_size, hidden_size, num_labels, x, y, learning_rate):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrxi(y)

    theta1 = np.matrix(np.reshape(
        params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(
        params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propgate(x, theta1, theta2)

    J = 0  # 初始化
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m

    for t in range(m):
        a1t = a1[t, :]
        z2t = z2[t, :]
        a2t = a2[t, :]
        ht = h[t, :]
        yt = y[t, :]

        d3t = ht-yt

        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T*d3t.T).T, S_gradient(z2t))

        delta1 = delta1+(d2t[:, 1:]).T*a1t
        delta2 = delta2+d3t.T*a2t
    delta1 = delta1/m
    delta2 = delta2/m

    return J, delta1, delta2

# 正则化神经网络


def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(
        params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(
        params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propgate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * \
        (np.sum(np.power(theta1[:, 1:], 2)) +
         np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, S_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))  # 将两个矩阵整合成一个矩阵

    return J, grad


fmin = minimize(fun=backpropReg, x0=(params), args=(input_size, hidden_size, num_labels,
                                                    x, y_onehot, learning_rate), method="TNC", jac=True, options={"maxiter": 250})

# 使用优化过后的X进行预测
x = np.matrix(x)
thetafinal1 = np.matrix(np.reshape(
    fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(
    fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propgate(x, thetafinal1, thetafinal2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

# 预测值与实际值比较
print(classification_report(y, y_pred))
