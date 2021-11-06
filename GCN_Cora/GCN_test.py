from sklearn.manifold import TSNE
import itertools
import os
import os.path as osp
import pickle
from collections import namedtuple
import numpy as np
from numpy.core.numeric import indices
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt

# 数据准备

Data = namedtuple("Data", ['x', "y", "adjacency",
                           'train_mask', "val_mask", "test_mask"])


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)  # 从numpy转为tensor


class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="./cora", rebuild=False):
        """Cora数据，包括，处理，加载等功能

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: ../data/cora
                缓存数据路径: {data_root}/ch5_cached.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "ch5_cached.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property  # 方法可以像属性一样调用
    def data(self):
        return self._data  # 返回Data数据对象

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("处理数据 ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        """
        tx 测试集1000个是论文给定，1433代表是有1433个特征点 (1000,1433)
        allx 训练集中的所有训练实例特征，包含有标签的和无标签的 (1708,1433)
        y 标签 140个训练集对应140个标签，7是总共有7种不同类型的标签(140,7)
        ty 测试集对应的标签 (1000,7)
        ally 是allx对应的标签，从1708-2707，共1000个 (1000,7) 半监督学习有的有标签有的无标签
        graph 格式{index: [index_of_neighbor_nodes]}，并且总共2708个，是所有节点的数目
        test_index 测试集的id索引
        """
        train_index = np.arange(y.shape[0])  # 训练集索引0-139
        val_index = np.arange(y.shape[0], y.shape[0]+500)  # 验证机索引140-140+500
        sorted_test_index = sorted(test_index)  # 默认升序排序

        x = np.concatenate((allx, tx), axis=0)  # shape(2708,1433)
        # argmax 沿轴返回最大值的索引。相当于是把标签为1的索引找了出来 shape(2708,)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]  # 将排序过后的重新赋值给x,y
        num_nodes = x.shape[0]  # 节点个数

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)  # 矩阵里面是布尔值
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True  # 将训练集、验证集、测试集掩码设置为True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod  # 返回函数的静态方法，不用实例化就可以调用
    def build_adjacency(adj_dict):
        """
        根据邻接表创建邻接矩阵
        """
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():  # 获取字典键 值
            # 相当于列表里面套了一个列表,构造双向边例如0-111,111-0
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])), shape=(
            num_nodes, num_nodes), dtype="float32")  # 创建邻接矩阵

        return adjacency

    @staticmethod
    def read_data(path):
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            # hasattr() 函数用于判断对象是否包含对应的属性。
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def normalization(adjacency):
        """
        正则化 L=D^-0.5 * (A+I) * D^-0.5
        """
        adjacency += sp.eye(adjacency.shape[0])  # A+I
        degree = np.array(adjacency.sum(1))  # 求度
        d_hat = sp.diags(np.power(degree, -0.5).flatten())  # flatten把矩阵变成一维矩阵
        return d_hat.dot(adjacency).dot(d_hat).tocoo()  # tocoo转换为稀疏矩阵


# 图卷积层定义
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        """
        首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        """
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)  # 添加参数bias到模型中
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)  # glorot的初始化权重
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adiacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adiacency, support)  # 稀疏矩阵相乘
        if self.use_bias:
            output += self.bias

        return output

    def __repr__(self):  # 类的实例化对象用来做“自我介绍”的方法
        return self.__class__.__name__+' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

# 模型定义


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


# 模型训练

LEARNING_RATE = 0.1
WEIGHT_DACAY = 5e-4
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 加载数据
dataset = CoraData().data
node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = tensor_from_numpy(node_feature, DEVICE)
tensor_y = tensor_from_numpy(dataset.y, DEVICE)
tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
tensor_val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
tensor_test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)
normalize_adjacency = CoraData.normalization(
    dataset.adjacency)  # 直接求出图滤波器~L，并且归一化处理

num_nodes, input_dim = node_feature.shape
indices = torch.from_numpy(np.asarray(
    [normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long()  # 设置索引

values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))  # 设置值
tensor_adjacency = torch.sparse.FloatTensor(
    indices, values, (num_nodes, num_nodes)).to(DEVICE)


# 模型定义
model = GcnNet(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask] #获取对应训练集的标签140
    for epoch in range(EPOCHS):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]  # 选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)  # k计算损失
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_acc, _, _ = test(tensor_train_mask)     # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)     # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history

# 测试函数


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1] #torch.max把索引和值分开来，[1]是获取值
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


loss, val_acc = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())

plot_loss_with_acc(loss, val_acc)

# 绘制测试数据的TSNE降维图
tsne = TSNE()
out = tsne.fit_transform(test_logits)
fig = plt.figure()
for i in range(7):
    indices = test_label == i
    x, y = out[indices].T
    plt.scatter(x, y, label=str(i))
plt.legend()
plt.show()