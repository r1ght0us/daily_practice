import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import numpy.random


X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y.reshape(
    X[0, :].shape), s=40, cmap=plt.cm.Spectral)  # plt.cm.Spectral不同样本生成不同的颜色
# plt.show()

# 数据集中有多少个训练示例？ 另外，变量“ X”和“ Y”的“shape”是什么？
shape_X = X.shape  # [2,400] 400个样本，每个样本2个特征
shape_Y = Y.shape

m = shape_X[1]  # 训练样本大小

# 简单的逻辑回归
clf = sklearn.linear_model.LogisticRegressionCV()  # 逻辑回归的分类器
clf.fit(X.T, Y.T)  # 根据给定的数据拟合模型
# 绘制决策边界
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("logistic regression")

LR_predictions = clf.predict(X.T)
# plt.show()
# print(float(np.dot(Y,LR_predictions)+np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) #准确率表示

# 神经网络模型
# 定义神经网络结构
'''
n_x：输入层的大小
n_h：隐藏层的大小（将其设置为4）
n_y：输出层的大小
'''


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)


# 初始化模型参数


def initialize_parameters(n_x, n_h, n_y):
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))
    W1 = rs.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = rs.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2}
    return parameters

# 正向传播


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


# 实现compute_cost（）以计算损失的值


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logprobs = Y*np.log(A2)+(1-Y)*np.log(1-A2)
    cost = -1/m*np.sum(logprobs)

    cost = np.squeeze(cost)  # 去除轴为1的维度

    return cost 

# 反向传播


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = 1/m*np.dot(dZ2, A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dW1 = 1/m*np.dot(dZ1, X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads

# 梯度下降


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

# 建立神经网络模型


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)  # 随机初始化参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        # 前向传播，返回A2也就是yhat，cache就是剩余参数
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)  # 代价函数
        grads = backward_propagation(parameters, cache, X, Y)  # 反向传播返回导数
        parameters = update_parameters(parameters, grads)  # 梯度下降更新参数

        if print_cost and i % 1000 == 0:
            print(i, cost)
    return parameters

def predict(parameters, X):
    predictions = []
    A2, cache = forward_propagation(X, parameters)
    for i in range(0, A2.shape[1]):
        if A2[0][i] > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.matrix(predictions)


parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))

parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=False)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#plt.title("Decision Boundary for hidden layer size " + str(4))

# predictions = predict(parameters, X)
# print('Accuracy: %d' % float((np.dot(Y, predictions.T) +
#                               np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

