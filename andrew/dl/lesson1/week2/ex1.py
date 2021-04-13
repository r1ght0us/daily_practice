# numpy 基础

# S函数和np.exp()

# 只能基于实数运算
import time
import numpy as np
import math


def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s

# 使用numpy实现sigmoid函数


def S(z):
    s = 1/(1+np.exp(-z))
    return s


#x = np.matrix([1, 2, 3])

# 创建函数sigmoid_grad（）计算sigmoid函数相对于其输入x的梯度


def S_gradient(x):
    s = S(x)
    ds = s*(1-s)

    return ds
#x = np.array([1, 2, 3])
# print(S_gradient(x))

# 重塑数组
# 实现image2vector() ,该输入采用维度为(length, height, 3)的输入，并返回维度为(length*height*3, 1)的向量


def image2vector(image):
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v


# image = np.array([[[0.67826139,  0.29380381],
#                    [0.90714982,  0.52835647],
#                    [0.4215251,  0.45017551]],

#                   [[0.92814219,  0.96677647],
#                    [0.85304703,  0.52351845],
#                    [0.19981397,  0.27417313]],

#                   [[0.60659855,  0.00533165],
#                    [0.10820313,  0.49978937],
#                    [0.34144279,  0.94630077]]])
# print(image2vector(image))

# 行标准化
# 执行 normalizeRows（）来标准化矩阵的行。 将此函数应用于输入矩阵x之后，x的每一行应为单位长度（即长度为1）向量
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm  # 广播问题？！
    return x


x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
# print(normalizeRows(x))

# 广播和softmax函数
# 用numpy实现softmax函数。 你可以将softmax理解为算法需要对两个或多个类进行分类时使用的标准化函数


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp/x_sum
    return s

# x = np.array([
#     [9, 2, 5, 0, 0],
#     [7, 5, 0, 0 ,0]])
# print(softmax(x))


# 向量化
# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
x1 = np.random.random_sample(1000)
x2 = np.random.random_sample(1000)
tic = time.process_time()  # 返回当前执行到这里的时间
dot = 0
for i in range(len(x1)):
    dot = dot+x1[i]*x2[i]
toc = time.process_time()
print(dot, 1000*(toc-tic))  # 性能太好了哭唧唧

tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i]*x2[i]
toc = time.process_time()
print(outer, 1000*(toc-tic))

tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print(mul, 1000*(toc-tic))

# np.multiply(),np.dot(),*三者的区别

# np.multiply() 矩阵对应元素位置相乘与@运算符相同作用
# np.dot() 执行矩阵乘法运算
# *
# 对数组执行对应位置相乘
# 对矩阵执行矩阵乘法运算

def L1(yhat,y):
    loss=np.sum(np.abs(y-yhat))
    return loss
def L2(yhat,y):
    loss=np.dot((y-yhat),(y-yhat).T)
    return loss