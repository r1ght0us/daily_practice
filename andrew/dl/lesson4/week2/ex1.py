# 残差网络
'''
作业需要实现：
    实现ResNets的基本构建块。
    将这些模块放在一起，以实现和训练用于图像分类的最新神经网络。
'''

from numpy.lib.function_base import select
import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
import torch.optim as optim
from resnets_utils import *

from torch.utils.data import DataLoader, sampler, TensorDataset

import torchvision.datasets as dset

import numpy as np
import h5py


class identity_block(nn.Module):  # 输入输出维度匹配
    def __init__(self, filters, in_channels_orig):
        super(identity_block, self).__init__()
        F1, F2 = filters
        self.conv2d_1 = nn.Conv2d(
            in_channels=in_channels_orig,
            out_channels=F1,
            kernel_size=1
        )
        # 数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.bn_1 = nn.BatchNorm2d(F1)  # num_features=(N,C,H,W)中的C
        self.relu_1 = nn.ReLU()

        self.conv2d_2 = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=3,
            padding=1  # (f-1)/2
        )
        self.bn_2 = nn.BatchNorm2d(F2)
        self.relu_2 = nn.ReLU()

        self.conv2d_3 = nn.Conv2d(  # bottleneck构建模块，使中间的3*3的filter数目不会影响下一层moudle
            in_channels=F2,
            out_channels=in_channels_orig,  # 假设使用标识块时，来自shortcut的输入和来自主路径的输入通道数量相同。
            kernel_size=1
        )
        self.bn_3 = nn.BatchNorm2d(in_channels_orig)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x_shortcut = x

        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x)

        x = x+x_shortcut

        x = self.relu_3(x)  # 实现图中所进行的捷径计算

        return x


class convolutional_block(nn.Module):  # 当输入与输出维度不相匹配时
    def __init__(self, filters, s, in_channels_orig):
        super(convolutional_block, self).__init__()

        F1, F2, F3 = filters

        self.conv2d_1 = nn.Conv2d(
            in_channels=in_channels_orig,
            out_channels=F2,
            kernel_size=1,
            stride=s
        )
        self.bn_1 = nn.BatchNorm2d(F1)
        self.relu_1 = nn.ReLU()

        self.conv2d_2 = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=3,
            padding=1
        )
        self.bn_2 = nn.BatchNorm2d(F2)
        self.relu_2 = nn.ReLU()

        self.conv2d_3 = nn.Conv2d(
            in_channels=F2,
            out_channels=F3,
            kernel_size=1
        )
        self.bn_3 = nn.BatchNorm2d(F3)

        self.conv2d_shortcut = nn.Conv2d(
            in_channels=in_channels_orig,
            out_channels=F3,
            kernel_size=1,
            stride=s  # 使得输入输出维度相匹配的关键值
        )
        self.bn_shortcut = nn.BatchNorm2d(F3)

        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x_shortcut = x

        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x)

        x_shortcut = self.conv2d_shortcut(x_shortcut)
        x_shortcut = self.bn_shortcut(x_shortcut)

        x = x+x_shortcut
        x = self.relu_3(x)

        return x


ResNet50 = nn.Sequential(
    nn.ConstantPad2d(3, 0),
    # 阶段1
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 阶段2
    convolutional_block([64, 64, 256], s=1, in_channels_orig=64),
    identity_block([64, 64], in_channels_orig=256),
    identity_block([64, 64], in_channels_orig=256),  # output_size=15*15*256
    # 阶段3
    convolutional_block([128, 128, 512], s=2, in_channels_orig=256),
    identity_block([128, 128], in_channels_orig=512),
    identity_block([128, 128], in_channels_orig=512),
    identity_block([128, 128], in_channels_orig=512),  # output_size=8*8*512
    # 阶段4
    convolutional_block([256, 256, 1024], s=2, in_channels_orig=512),
    identity_block([256, 256], in_channels_orig=1024),
    identity_block([256, 256], in_channels_orig=1024),
    identity_block([256, 256], in_channels_orig=1024),
    identity_block([256, 256], in_channels_orig=1024),
    identity_block([256, 256], in_channels_orig=1024),  # output_size=4*4*1024
    # 阶段5
    convolutional_block([512, 512, 2048], s=2, in_channels_orig=1024),
    identity_block([256, 256], in_channels_orig=2048),
    identity_block([256, 256], in_channels_orig=2048),  # output_size=2*2*2048


    nn.AvgPool2d(kernel_size=2, stride=1),  # output_size=1*1*2048
    nn.Flatten(),  # 展平成一个张量
    nn.Linear(2048, 6)
)

# nn.LogSoftmax()和nn.NLLLoss()两个函数的结合，先进行logsoftmax然后NLLLoss
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(ResNet50.parameters(), lr=1e-3)


X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()
X_train_orig = np.transpose(X_train_orig, (0, 3, 1, 2))
X_test_orig = np.transpose(X_test_orig, (0, 3, 1, 2))

Y_train = Y_train.ravel()
Y_test = Y_test.ravel()

X_train = X_train_orig/255
X_test = X_test_orig/255

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)


train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
loader_test = DataLoader(test_dataset, batch_size=32, shuffle=True)


def get_accuracy(model, loader_test): #计算准确率
    '''
    在正常建立好模型运行程序时，module都必须先要train,所以总是在训练状态
    但如果是下载好别人训练好的模型而不需要自己训练时，运行程序就应该变为eval()模式。
    '''
    model.eval()
    num_samples, num_correct = 0, 0
    with torch.no_grad(): #不进行计算，不需要保存梯度
        for x, y in loader_test:
            output = model(x)
            _, y_pred = output.data.max(1) #y_red返回索引
            num_correct += (y_pred == y).sum().item() #item()从张量返回常数
            num_samples += x.size(0) #样本数

    return num_correct/num_samples


def train(model, loss_fn, optimizer, loader_train, loader_test, epochs=1):
    for epoch in range(epochs):
        model.train() #训练模式
        for i, (x, y) in enumerate(loader_train):
            y_pred = model(x)

            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()
        acc = get_accuracy(model, loader_test)
        print(f"Epoch: {epoch+1} | Loss: {loss.item()} | Test accuracy: {acc}")

train(ResNet50, loss_fn, optimizer, loader_train, loader_test, epochs=5)
