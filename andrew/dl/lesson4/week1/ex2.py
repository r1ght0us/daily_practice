import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from cnn_utils import load_dataset

torch.manual_seed(1)  # 设置随机数种子

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

'''
#可视化一个样本

index=4
plt.imshow(X_train_orig[index])

print(np.suqeeze(Y_train_orig[:,index]))
plt.show()
'''

# np.transpose将维度从(1080,64,64,3)转换为(1080,3,64,64)
X_train = np.transpose(X_train_orig, (0, 3, 1, 2))/255
X_test = np.transpose(X_test_orig, (0, 3, 1, 2))/255  # 将维度转为(120, 3, 64, 64)

Y_train = Y_train_orig.T  # (1080,1)
Y_test = Y_test_orig.T  # (120,1)


def data_loader(X_train, Y_train, batch_size=64):
    '''
    创建数据接口
    '''
    train_db = TensorDataset(torch.from_numpy(
        X_train).float(), torch.squeeze(torch.from_numpy(Y_train)))
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)

    return train_loader


# 构建模型
'''
一个 module 里可包含多个子 module。比如 LeNet 是一个 Module，里面包括多个卷积层、池化层、全连接层等子 module
'''


class CNN(nn.Module):  # CNN继承nn.Moudle类
    def __init__(self):
        super(CNN, self).__init__()  # 父类初始化

        self.conv1 = nn.Sequential(  # 输入维度3*64*64
            nn.Conv2d(  # 应用二维卷积
                in_channels=3,  # 输入图像的通道数
                out_channels=8,  # 卷积之后产生的通道数
                kernel_size=4,  # 卷积核的大小
                stride=1,  # 步长
                padding=1  # p=(f-1)/2 f=4然后p需要向下取整
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8, padding=4)
            # 应用2维最大池化层，
            # 8*8的过滤器
            # 步幅为8
            # 为“same”填充（不清楚为啥）
        )

        self.conv2 = nn.Sequential(  # 输入维度8*8*8
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=2,
                stride=1,
                padding=1  # padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=2)
            # output shape:16*3*3
        )

        self.fullconnect = nn.Sequential(
            nn.Linear(in_features=16*3*3, out_features=6),
            # nn.ReLU()
        )

        self.classifier = nn.LogSoftmax(dim=1)  # 对列进行softmax

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)  # 展示成一维张量
        x = self.fullconnect(x)
        output = self.classifier(x)

        return output


def weight_init(m):  # 初始化权重以及偏置
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)  # xavier初始化
        nn.init.constant_(m.bias, 0)  # 定义张量为定值
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=150, minibatch_size=64, print_cost=True, is_plot=True):
    train_loader = data_loader(X_train, Y_train, minibatch_size)

    cnn = CNN()
    cnn.apply(weight_init)  # apply()经常用于初始化参数

    cost_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        cnn.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    costs = []

    m = X_train.shape[0]
    num_batch = m/minibatch_size  # 总共有多少个批量
    
    for epoch in range(num_epochs):
        epoch_cost = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x) #前向传播

            cost = cost_func(output, batch_y)
            epoch_cost = epoch_cost+cost.data.numpy()/num_batch #统计一个epoch的cost

            optimizer.zero_grad() 

            cost.backward()

            optimizer.step()
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)
            print(epoch, epoch_cost)

    if is_plot:
        plt.plot(costs)
        plt.xlabel("iteration per 5")
        plt.ylabel("cost")
        plt.show()

    torch.save(cnn.state_dict(), "net_params.pkl")
    
    cnn.load_state_dict(torch.load("net_params.pkl"))
    output_train = cnn(torch.from_numpy(X_train).float())  # 加载保存好的参数后，测试集输出
    pred_Y_train = torch.max(output_train, dim=1)[
        1].data.numpy()  # dim=1表示成同一行的每一列进行比较,官方文档中表示这个是你要缩减的维度（行不变列要变，那么肯定是对列进行操作）
                         #获得每行最大的索引就相当于获得预测输出是多少

    output_test = cnn(torch.from_numpy(X_test).float())
    pred_Y_test = torch.max(output_test, dim=1)[1].data.numpy()

    print('Train Accuracy: %.2f %%' %
          float(np.sum(np.squeeze(Y_train) == pred_Y_train)/m*100))

    print('Test Accuracy: %.2f %%' %
          float(np.sum(np.squeeze(Y_test) == pred_Y_test)/X_test.shape[0]*100))

    return cnn


model(X_train, Y_train, X_test, Y_test)
