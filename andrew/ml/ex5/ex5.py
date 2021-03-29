# 主要知识点为偏差和方差，训练集&验证集&测试集

# 正则化线性回归
# 需要先对一个水库的流出水量以及水库水位进行正则化线性归回

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sio.loadmat("ex5data1.mat")
x, y, xval, yval, xtest, ytest = map(np.ravel, [
                                     data["X"], data["y"], data["Xval"], data["yval"], data["Xtest"], data["ytest"]])  # map函数：第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。

# 数据可视化
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.scatter(x, y)
# ax.set_xlabel("water_level")
# ax.set_ylabel("flow")
# plt.show()

# 正则化线性回归代价函数
x, xval, xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(
    x.shape[0]), axis=1) for x in (x, xval, xtest)]  # 为x xval xtest添加常数项


def cost(theta, x, y):
    m = x.shape[0]
    inner = x@theta-y
    square_sum = inner.T@inner  # 这一步进行了平方合求和，太妙了!!
    cost = square_sum/(2*m)
    return cost


def costREG(theta, x, y, reg=1):
    m = x.shape[0]
    regularized_term = (reg/(2*m))*(np.power(theta[1:], 2).sum())
    return cost(theta, x, y)+regularized_term

# 正则线性回归的梯度


def gradient(theta, x, y):
    m = x.shape[0]
    inner = x.T@(x@theta-y)
    return inner/m


def gradientREG(theta, x, y, reg):
    m = x.shape[0]
    regularized_term = theta.copy()  # 返回一个复制
    regularized_term[0] = 0  # theta_0不进行梯度
    regularized_term = (reg/m)*regularized_term
    return gradient(theta, x, y)+regularized_term


# 拟合线性回归
# 调用工具库找到theta最优解，在这个部分，我们令正则化=0。因为我们现在训练的是2维的，所以正则化不会对这种低维的theta有很大的帮助。完成之后，将数据和拟合曲线可视化。


theta = np.ones(x.shape[1])
final_theta = opt.minimize(fun=costREG, x0=theta, args=(
    x, y, 0), method="TNC", jac=gradientREG, options={"disp": False}).x  # disp设置True将打印出可聚合信息

b = final_theta[0]  # 截距
m = final_theta[1]  # 斜率

# fig, ax = plt.subplots(figsize=(8, 5))
# plt.scatter(x[:, 1], y, c="r", label="training data")
# plt.plot(x[:, 1], x[:, 1]*m+b, c="b", label="prediction")  # 画出预测直线
# ax.set_xlabel("water_level")
# ax.set_ylabel("flow")
# ax.legend()
# plt.show()

# 方差与偏差
# 学习曲线
# 1.使用训练集的子集来拟合应模型
# 2.在计算训练代价和验证集代价时，没有用正则化
# 3.记住使用相同的训练集子集来计算训练代价


def linear_regression(x, y, l=1):
    theta = np.ones(x.shape[1])

    res = opt.minimize(fun=costREG, x0=theta, args=(
        x, y, l), method="TNC", jac=gradientREG, options={"disp": False})
    return res


training_cost = []
cv_cost = []

m = x.shape[0]
for i in range(1, m+1):
    res = linear_regression(x[:i, :], y[:i], 0)
    tc = costREG(res.x, x[:i, :], y[:i], 0)  # 模拟随着数据集增大进行的代价函数计算
    cv = costREG(res.x, xval, yval, 0)

    training_cost.append(tc)
    cv_cost.append(cv)

# fig, ax = plt.subplots(figsize=(8, 5))
# plt.plot(np.arange(1, m+1), training_cost, label="training cost")
# plt.plot(np.arange(1, m+1), cv_cost, label="cv cost")
# plt.legend()
# plt.show()


# 多项式回归
# 写一个函数，输入原始X，和幂的次数p，返回X的1到p次幂
def poly_features(x, power, as_ndarray=False):  # 扩展到8阶特征
    data = {"f{}".format(i): np.power(x, i) for i in range(1, power+1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


x, y, xval, yval, xtest, ytest = map(np.ravel, [
                                     data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])
# 使用之前的代价函数和梯度函数
# 扩展特征到8阶特征
# 使用 归一化 来处理 x^n
# λ=0


def normalize_feature(df):  # 特征缩放，保证这些特征都具有相同的尺度
    # lamda匿名函数
    # apply函数是将每一列的数据应用到funs函数中
    # mean函数是求平均值
    # std函数返回标准差
    return df.apply(lambda column: (column-column.mean())/column.std())


def prepare_poly_data(*args, power):
    # *args 位置传参
    def prepare(x):
        df = poly_features(x, power=power)
        ndarr = normalize_feature(df).values

        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)
    return [prepare(x) for x in args]


x_poly, xval_poly, xtest_poly = prepare_poly_data(x, xval, xtest, power=8)

# 绘制学习曲线


def plot_learning_curve(x, xinit, y, xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = x.shape[0]

    for i in range(1, m+1):
        res = linear_regression(x[:i, :], y[:i], l=l)  # 进行正则化
        tc = cost(res.x, x[:i, :], y[:i])
        cv = cost(res.x, xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(np.arange(1, m+1), training_cost, label="training cost")
    ax[0].plot(np.arange(1, m+1), cv_cost, label="cv cost")
    ax[0].legend()

    fitx = np.linspace(-50, 50, 100)
    fity = np.dot(prepare_poly_data(fitx, power=8)[
                  0], linear_regression(x, y, l).x.T)  # 点积相乘

    ax[1].plot(fitx, fity, c="r", label="fitcurve")
    ax[1].scatter(xinit, y, c="b", label="initial_xy")
    ax[1].legend()
    ax[1].set_xlabel("water_level")
    ax[1].set_ylabel("flow")


# 找到最佳的λ

l_candidate = [0, 00.1, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression(x_poly, y, l)

    tc = cost(res.x, x_poly, y)
    cv = cost(res.x, xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(l_candidate, training_cost, label='training')
# ax.plot(l_candidate, cv_cost, label='cross validation')
# plt.legend()
# plt.xlabel('lambda')
# plt.ylabel('cost')
# plt.show()

# 计算测试集的误差
for l in l_candidate:
    theta = linear_regression(x_poly, y, l).x
    print("test cost(l={0})={1}".format(l, cost(theta, xtest_poly, ytest)))
