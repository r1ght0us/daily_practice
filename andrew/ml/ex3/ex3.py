import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report
from scipy.optimize import minimize

# 多分类

# 数据可视化
data = loadmat("ex3data1.mat")
rng = np.random.default_rng()  # 调用default_rng以获取Generator的新实例
sample_idx = rng.choice(data['X'].shape[0], 100)
# 此方法用于生成给定的序列中随机抽取n个数，此行的作用是将第一个参数的序列随机抽取100个数字。如果第一个参数是INT那么等同于 np.arange(a) （生成从0-a的等差为1的序列）
sample_images = data["X"][sample_idx, :]

fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True,
                             sharex=True, figsize=(12, 8))  # 生成10x10网格图，子图共享x轴和y轴
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(
            np.array(sample_images[10*r+c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
        # matshow将数组或者二维矩阵转换为图像，cmap表示图像标记
        # reshape将数组转换成20*20矩阵
        plt.xticks(np.array([]))  # 删除所有x轴上的标记
        plt.yticks(np.array([]))

# 逻辑回归向量化

# S函数


def S(z):
    return 1/(1+np.exp(-z))

# 代价函数


def cost(theta, x, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(S(x*theta.T)))
    second = np.multiply((1-y), np.log(1-S(x*theta.T)))
    reg = (learningRate/(2*len(x))) * \
        np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first-second)/len(x)+reg

# 向量化梯度


def gradient(theta, x, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    error = S(x*theta.T)-y
    grad = ((x.T*error)/len(x)).T+((learningRate/len(x))*theta)

    grad[0, 0] = np.sum(np.multiply(error, x[:, 0]))/len(x)

    return np.array(grad).ravel()


# 一对多分类器
def one_vs_all(x, y, num_labels, learning_rate):
    rows = x.shape[0]
    params = x.shape[1]
    # 该函数计算10个分类器中的每个分类器的最终权重，并将权重返回为k*(n + 1)数组，其中n是参数数量。
    all_theta = np.zeros((num_labels, params+1))
    x = np.insert(x, 0, values=np.ones(rows), axis=1)  # 插入初始列

    for i in range(1, num_labels+1):
        theta = np.zeros(params+1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(
            x, y_i, learning_rate), method="TNC", jac=gradient)
        # fun 求最小值的目标函数
        # x0初始自变量参数
        # args传递给函数的参数
        # jac梯度计算的方法
        all_theta[i-1, :] = fmin.x
    return all_theta


# 写入所有矩阵
rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10, params+1))
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params+1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

all_theta = one_vs_all(data["X"], data["y"], 10, 1)

# 一对多预测


def predict_all(X, all_theta):
    rows = X.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    h = S(X*all_theta.T)
    h_argmax = np.argmax(h, axis=1)  # argmax,axis=1返回每一行的最大值的索引值
    h_argmax = h_argmax+1  # 为真正的标签+1
    return h_argmax


y_pred = predict_all(data["X"], all_theta)
# print(classification_report(data['y'],y_pred)) 精确率，召回率，F1

# 神经网络

# 前馈神经网络和预测
weight = loadmat("ex3weights.mat")
theta1, theta2 = weight["Theta1"], weight["Theta2"]

# 插入常数项
x2 = np.matrix(np.insert(data["X"], 0, values=np.ones(X.shape[0]), axis=1))
y2 = np.matrix(data["y"])

a1 = x2
z2 = a1*theta1.T
a2 = S(z2)
a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
z3 = a2*theta.T
a3=S(z3)
