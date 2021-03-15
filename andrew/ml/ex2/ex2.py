import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# 展示前五行数据
path = "ex2data1.txt"
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
# print(data.head())

# # 绘制原始数据
positive = data[data['admitted'].isin([1])]  # 筛选admitted列中是1的数据出来
negative = data[data['admitted'].isin([0])]

# fig, ax = plt.subplots(figsize=(8, 5))  # 设置出来的画布大小为8*5
# # 绘制散点图。x轴为exam1，y轴为label2，颜色是绿色，标记是o（圆形），绘制的标签是admitted
# ax.scatter(positive["exam1"], positive["exam2"],
#            c='green', marker='o', label="admitted")
# # 绘制散点图。x轴为exam1，y轴为label2，颜色是红色，标记是x（x形），绘制的标签是not admitted
# ax.scatter(negative["exam1"], negative["exam2"],
#            c='red', marker='x', label="not admitted")
# ax.legend()  # 绘制标签将其显示出来
# # plt.show()

# 实现

# S型函数（Sigmoid）


def S(z):
    return 1/(1+np.exp(-z))

# 代价函数


def cost(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(S(x*theta.T)))
    second = np.multiply((1-y), np.log(1-S(x*theta.T)))
    return np.sum(first-second)/len(x)


# 初始化操作
data.insert(0, "Ones", 1)  # 在data中的第0列插入表头为ONEs数据为1的一列数据
cols = data.shape[1]  # 返回data矩阵的形式，0代表行，1代表列
x = data.iloc[:, 0:cols-1]  # 基于位置的索引
y = data.iloc[:, cols-1:cols]
theta = np.zeros(3)

# x.values文档中推荐使用x.to_numpy()代替同样可以返回numpy.ndarray数据
x = np.array(x.to_numpy())
y = np.array(y.to_numpy())

# 梯度下降


def gradient(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # ravel()是将其矩阵化为1维矩阵
    grad = np.zeros(parameters)

    error = S(x*theta.T)-y
    for i in range(parameters):
        term = np.multiply(error, x[:, i])
        grad[i] = np.sum(term)/len(x)

    return grad


# 使用工具库计算θ的值

# result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
# 截断牛顿法的传参列表：
# func：优化的目标函数
# x0：初值
# fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度，或者设置approx_grad=True
# approx_grad :如果设置为True，会给出近似梯度
# args：元组，是传递给优化函数的参数

# 返回值是第一个为数组，返回的优化问题目标值相当于返回θ。第二个整数，优化函数运行次数

# 画出决策曲线
# plotting_x1 = np.linspace(30, 100, 100)  # 30-100中平均划分100个数字
# plotting_h1 = (-result[0][0]-result[0][1]*plotting_x1)/result[0][2]

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(plotting_x1, plotting_h1, color='black', label='prediction')
# ax.scatter(positive["exam1"], positive["exam2"],
#            c='green', marker='o', label="admitted")
# ax.scatter(negative["exam1"], negative["exam2"],
#            c='red', marker='x', label="not admitted")
# ax.legend()
# ax.set_xlabel("exam1 score")
# ax.set_ylabel("exam2 score")
# plt.show()

# 评价逻辑回归模型
# 手动模拟一个数据


def hfunc(theta, x):
    return S(theta.T@x)  # 疑问：输入[1,0,0]得到的录取率180%多是否做边界检查?


def predict(theta, x):
    predictions = []
    probability = S(x@theta.T)
    for x in probability:
        if x >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# 使用训练集评测模型的准确率
# theta_min = np.matrix(result[0])
# predictions = predict(theta_min, x)
# corrent = []
# print(type(predictions), type(y))
# for (a, b) in zip(predictions, y):
#     if (a == 1 and b == 1) or (a == 0 and b == 0):
#         corrent.append(1)
#     else:
#         corrent.append(0)
# accuracy = (sum(corrent))/len(corrent)
# print("accuracy is {0}%".format(accuracy*100))

# 正则化逻辑回归

# 数据可视化
path = "ex2data2.txt"
data2_init = pd.read_csv(path, header=None, names=[
                         "test1", "test2", "accepted"])
positive2 = data2_init[data2_init["accepted"].isin([1])]
negative2 = data2_init[data2_init["accepted"].isin([0])]
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.scatter(positive2["test1"], positive2["test2"],
#            c="b", marker="o", label="accepted")
# ax.scatter(negative2["test1"], negative2["test2"],
#            c="r", marker="x", label="rejected")
# ax.legend()
# ax.set_xlabel("test1 score")
# ax.set_ylabel("test2 score")
# plt.show()

# 特征映射
degree = 6
data2 = data2_init
x1 = data2["test1"]
x2 = data2["test2"]

data2.insert(3, "ones", 1)

for i in range(1, degree+1):
    for j in range(0, i+1):
        data2["f"+str(i-j)+str(j)] = np.power(x1, i-j)*np.power(x2, j)

# 删除test1这一列,axis=1代表删除的是列，inplace就地删除并且不放回所要删除的东西
data2.drop("test1", axis=1, inplace=True)
data2.drop("test2", axis=1, inplace=True)
# print(data2.head())

# 代价函数和梯度

# 正则化代价函数


def costREG(theta, x, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(S(x*theta.T)))
    second = np.multiply((1-y), np.log(1-S(x*theta.T)))
    reg = (learningRate/(2*len(x))) * \
        np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first-second)/len(x)+reg


# 正则化梯度函数
def gradientREG(theta, x, y, learningRate):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = S(x*theta.T)-y

    for i in range(parameters):
        term = np.multiply(error, x[:, i])

        if i == 0:
            grad[i] = np.sum(term)/len(x)
        else:
            grad[i] = (np.sum(term)/len(x))+((learningRate/len(x))*theta[:, i])
    return grad


# 初始化x,y,theta
cols = data2.shape[1]
x2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]
theta2 = np.zeros(cols-1)
x2 = np.array(x2.to_numpy())
y2 = np.array(y2.to_numpy())

# 布兰达设置为1
learningRate = 1
result2 = opt.fmin_tnc(func=costREG, x0=theta2,
                       fprime=gradientREG, args=(x2, y2, learningRate))

# 使用训练集评测模型的准确率
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, x2)

corrent = []
for (a, b) in zip(predictions, y2):
    if (a == 1 and b == 1) or (a == 0 and b == 0):
        corrent.append(1)
    else:
        corrent.append(0)
accuracy = (sum(corrent))/len(corrent)
# print("accuracy is {0}%".format(accuracy*100))


def hfunc2(theta, x1, x2):
    temp = theta[0][0]
    place = 0
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp = temp + np.power(x1, i-j)*np.power(x2, j)*theta[0][place+1]
            place = place+1
    return temp


def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({"x1": x_cord, "x2": y_cord})
    h_val["hval"] = hfunc2(theta, h_val["x1"], h_val["x2"])

    decision = h_val[np.abs(h_val["hval"]) < 2*10**-3]
    return decision.x1, decision.x2

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive2['test1'], positive2['test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['test1'], negative2['test2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()