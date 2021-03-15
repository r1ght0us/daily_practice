import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 输出一个5*5的对角矩阵
A = np.eye(5)
# print(A)

# 单变量的线性回归

## 展示数据

path = "C:\\Users\\r1ght0us\\Desktop\\andrew\\ml\\ex1\\ex1data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())  #head()只是选择数据，要用print输出
# 画一个散点图（scatter plot），X轴为人口，Y轴是利润，图像大小是12*8
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
#plt.show()  # 展示图像

## 梯度下降

### 初始化
data.insert(0, 'Ones', 1)  # 插入一列用于更新 θ_0，方便向量化才做如此操作
cols = data.shape[1] #返回data中的矩阵大小，返回形式是以元组形式返回
x = data.iloc[:, :-1]  # X是data里的除最后列的剩余列，取x值
y = data.iloc[:, cols-1:cols]  # y是data最后一列,相当于取代价函数的y值

x=np.matrix(x.to_numpy())# 转换成numpy识别的形式
y=np.matrix(y.to_numpy())
theta=np.matrix(np.array([0,0])) #数组转换为矩阵

### 代价函数
def FunctionOfCost(x,y,theta):
    inner = np.power((x*theta.T-y),2) # theta.T是返回矩阵的转置，power就是平方
    return np.sum(inner)/(2*len(x)) #sum计算总和

### 梯度下降
def gradientDescent(x,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape)) # 若干特征的大小矩阵
    parameters = int(theta.ravel().shape[1]) # 每次迭代的θ_n存放的地方
    cost = np.zeros(iters) #每次迭代得到的代价函数值

    for i in range(iters): # 迭代过程
        error = x*theta.T-y # 初始的代价函数值

        for j in range(parameters):
            term = np.multiply(error,x[:,j]) #代价函数值与x^(i)相乘
            temp[0,j]=theta[0,j] - ((alpha/len(x))*np.sum(term)) #求出来的θ分别赋值
        
        theta = temp
        cost[i] = FunctionOfCost(x,y,theta) 
    
    return theta,cost

alpha = 0.01 #学习率
iters = 1500 #迭代次数
g, cost = gradientDescent(x, y, theta, alpha, iters)

## 画出拟合之后的图
# a = np.linspace(data.Population.min(), data.Population.max(), 100)# 在最小值和最大值之间均匀的风格100个数字
# f = g[0, 0] + (g[0, 1] * a) # 预测出来的线性回归函数

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(a, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()

## 代价函数图像
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()

# 多变量线性回归
path2 =  'C:\\Users\\r1ght0us\\Desktop\\andrew\\ml\\ex1\\ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])

## 特征缩放((原值-平均值)/标准差)
data2 = (data2 - data2.mean()) / data2.std()

## 初始化
data2.insert(0, 'Ones', 1)

cols = data2.shape[1]
x2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

x2=np.matrix(x2.to_numpy())# 转换成numpy识别的形式
y2=np.matrix(y2.to_numpy())
theta2 = np.matrix(np.array([0,0,0]))

## 梯度下降
g2, cost2 = gradientDescent(x2, y2, theta2, alpha, iters)

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y # @为矩阵相乘
    return theta

final_theta2=normalEqn(x, y)
print(final_theta2)
print(g)
