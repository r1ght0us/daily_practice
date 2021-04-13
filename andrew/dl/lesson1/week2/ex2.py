# 用神经网络思想实现Logistic回归

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

'''
数据集包含以下内容：
1. 标记为cat（y = 1）或非cat（y = 0）的m_train训练图像集
2. 标记为cat或non-cat的m_test测试图像集
3. 图像维度为（num_px，num_px，3），其中3表示3个通道（RGB）。 因此，每个图像都是正方形（高度= num_px）和（宽度= num_px）
'''
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# train_set_x_orig和test_set_x_orig的每一行都是代表图像的数组
#index = 5
# plt.imshow(train_set_x_orig[index])#将array-like或者PIL image转换成图像
# plt.show()
# print(train_set_y[:,index],classes[np.squeeze(train_set_y[:,index])].decode("utf-8"))


#  重塑训练和测试数据集，以便将大小（num_px，num_px，3）的图像展平为单个形状的向量(num_px  num_px  3, 1)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 对数据集进行标准和处理化，意味着你要从每个示例中减去整个numpy数组的均值，然后除以整个numpy数组的标准差。但是图片数据集则更为简单方便，并且只要将数据集的每一行除以255（像素通道的最大值），效果也差不多。

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# 使用神经网络建立逻辑回归

# 构建算法的各个部分
# S函数


def S(z):
    s = 1/(1+np.exp(-z))
    return s

# 初始化参数


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    # isinstance函数来判断一个对象是否是一个已知的类型，类似 type()。
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# dim=2
# w,b=initialize_with_zeros(dim)

# 前向传播和后向传播
# 实现函数propagate（）来计算损失函数及其梯度


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = S(np.dot(w.T, X)+b)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    dw = 1/m*np.dot(X, (A-Y).T)
    db = 1/m*np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    assert(cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost


w, b, X, Y = np.array([[1], [2]]), 2, np.array(
    [[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)

# 优化函数
'''
初始化参数。
计算损失函数及其梯度。
使用梯度下降来更新参数。
'''


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []  # 绘制学习曲线

    for i in range(num_iterations):  # 迭代100次
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


params, grads, costs = optimize(
    w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

# 实现预测函数
# 上一个函数将输出学习到的w和b。 我们能够使用w和b来预测数据集X的标签。实现predict（）函数。 预测分类有两个步骤：
'''
计算预测Y
将a的项转换为0（如果激活<= 0.5）或1（如果激活> 0.5），并将预测结果存储在向量“ Y_prediction”中
'''


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = S(np.dot(w.T, X)+b)  # 使用训练好的w即特征集合进行预测

    for i in range(A.shape[1]):
        if A[0, i] < 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


# 将所有功能合并到模型
'''
Y_prediction对测试集的预测
Y_prediction_train对训练集的预测
w，损失，optimize（）输出的梯度
'''


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]#获取训练好的w,b

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w, "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=False)

#绘制学习曲线

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()