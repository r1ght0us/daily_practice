import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tf_utils import load_dataset

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))

# y_hat = torch.tensor(36)
# y = torch.tensor(39)

# loss = (y-y_hat)**2

# loss = loss.numpy()
# print(loss)

# a=torch.tensor(2)
# b=torch.tensor(10)
# #对比：pytorch的元素乘法使用torch.mul()或 * ，numpy使用np.multiply()或 * ，都有广播机制。
# c=torch.multiply(a,b).numpy()
# print(c)

# #张量内积
# d=torch.tensor([2,2])
# e=torch.tensor([1,4])
# print(torch.dot(d,e).numpy())  #print 10
# #矩阵乘法
# f=torch.tensor([[1],[4]])
# print(torch.matmul(d,f).numpy())#print [10]

# 线性函数
# def linear_function():
#     X=torch.from_numpy(rs.randn(3,1))
#     W=torch.from_numpy(rs.randn(4,3))
#     b=torch.from_numpy(rs.randn(4,1))
#     Y=torch.add(torch.matmul(W, X), b) #两个张量相加

#     return Y.numpy()

# print(linear_function())


def sigmoid(z):
    z = torch.tensor(z, dtype=torch.float32)
    sig = torch.sigmoid(z)  # torch.sigmoid()函数只接受张量
    return sig

# print(sigmoid(0).numpy())
# print(sigmoid(10).numpy())

# 代价函数


def cost(logits, labels):
    input = sigmoid(logits.numpy())

    loss = nn.BCELoss(reduction="none")  # reduction默认是求平均值
    cost = loss(input, labels)

    return cost

# logits=sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
# labels = torch.Tensor([0, 0, 1, 1])
# cost = cost(logits, labels)
# print('cost=', cost.numpy())

# 独热编码


def one_hot(label, C):
    m = labels.shape[0]
    # 将独热矩阵初始化为全0，torch.zeros与np.zeros类似，torch建立的是tensor类型，而np建立的是ndarry类型
    one_hot_matrix = torch.zeros(size=(C, m))

    for l in range(m):
        one_hot_matrix[label[l]][l] = 1

    return one_hot_matrix.numpy()

# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot(labels, C=4)
# print(one_hot)

# 使用pytorch构建第一个神经网络


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 可视化一张图片
# index = 4
# plt.imshow(X_train_orig[index])
# plt.show()
# print('y = ', Y_train_orig[:, index])

# 展开数据集
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1)
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1)

# 归一化数据集
X_train = X_train_flatten/255
X_test = X_test_flatten/255
# 转置y
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# 创建网络结构


def nerual_net():
    net = nn.Sequential(nn.Linear(12288, 25), nn.ReLU(), nn.Linear(
        25, 12), nn.ReLU(), nn.Linear(12, 6), nn.LogSoftmax(dim=1))  # 顺序容器。 模块将按照在构造函数中传递的顺序添加到模块中
    return net

# 初始化参数


def initialize_params(net):
    for i in range(len(net)):
        layer = net[i]
        if isinstance(layer, nn.Linear):
            # 对wi使用xavier初始化
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(
                "relu"))  # nn.init.calculate_gain:返回给定非线性函数的推荐增益值
            # 对bi使用0初始化
            nn.init.constant_(layer.bias, 0)  # 以0填充偏置单元

    return net

# 这个函数将X_train, Y_train打包，可以理解为打包成类似于小批量迭代器的形式，每历遍所有批次再进行下一轮迭代，它都会自动打乱数据集，使每个批次内的样本保持随机。


def data_loader(X_train, Y_train, batch_size=32):
    train_db = TensorDataset(torch.from_numpy(
        X_train).float(), torch.squeeze(torch.from_numpy(Y_train)))
    # batch_size表示每个batch大小是多少，shuffle随机排序
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)

    return train_loader


def model(X_train, Y_train, X_test, Y_test, lr=0.0001, epochs=1500, batch_size=32, print_cost=True, is_plot=True):

    # 载入数据
    train_loader = data_loader(X_train, Y_train, batch_size)
    # 创建网络结构
    net = nerual_net()
    # 指定成本函数
    cost_func = nn.NLLLoss()  # 负对数似然估计
    # 指定优化器为adam
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(
        0.9, 0.999))  # net.parameters()获取到神经网络中所用到的权重、 偏置

    costs = []

    m = X_train.shape[0]
    num_batch = m/batch_size  # 统计有多少个patch

    #net = initialize_params(net)

    for epoch in range(epochs):
        epoch_cost = 0

        for step, (batch_x, batch_y) in enumerate(train_loader):
            # 前向传播
            output = net(batch_x)
            # 计算成本
            # torch.squeeze 将输入张量形状中的1去除并返回。
            cost = cost_func(output, torch.squeeze(batch_y))
            epoch_cost = epoch_cost+cost.data.numpy()/num_batch
            # 梯度归零
            optimizer.zero_grad()  # 将module中的所有模型参数的梯度设置为0.
            # 反向传播
            cost.backward()
            # 更新参数
            optimizer.step()  # 这个方法会更新所有的参数
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)
            if epoch % 100 == 0:
                print(epoch, epoch_cost)

    if is_plot:
        plt.plot(costs)
        plt.xlabel("iterations per 5")
        plt.ylabel("cost")
        plt.show()

    # 保存学习后的参数
    # state.dict() 返回一个字典，保存着module的所有状态
    torch.save(net.state_dict(), "net_params.pkl")
    # 计算训练集预测结果
    # 将state_dict中的parameters和buffers复制到此module和它的后代中
    net.load_state_dict(torch.load("net_params.pkl"))
    output_train = net(torch.from_numpy(X_train).float())  # 将训练好的模型套在训练集
    test = torch.max(output_train, dim=1)
    # torch.max()返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引
    pred_Y_train = torch.max(output_train, dim=1)[1].data.numpy()
    # 计算测试集预测的结果
    output_test = net(torch.from_numpy(X_test).float())
    pred_Y_test = torch.max(output_test, dim=1)[1].data.numpy()

    print('Train Accuracy: %.2f %%' %
          float(np.sum(np.squeeze(Y_train) == pred_Y_train)/m*100))
    print('Test Accuracy: %.2f %%' %
          float(np.sum(np.squeeze(Y_test) == pred_Y_test)/X_test.shape[0]*100))

    return net


model(X_train, Y_train, X_test, Y_test, lr=0.0001, epochs=5,
      batch_size=32, print_cost=True, is_plot=True)
