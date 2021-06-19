# 字符语言模型
# 此代码中h代表hidden隐藏层=RNN中的传递的激活值a
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.utils import softmax

torch.set_printoptions(linewidth=200)

hidden_size = 100


class DinosDataset(Dataset):  # Dataset是抽象类，所有其他数据集都应该进行子类化
    def __init__(self):
        super().__init__()
        with open("data/dinos.txt") as f:
            content = f.read().lower()
            # set() 函数创建一个无序不重复元素集,sorted()进行排序
            self.vocab = sorted(set(content))
            self.vocab_size = len(self.vocab)  # 包含换行符
            self.lines = content.splitlines()  # 返回以每行作为切割的列表

        self.ch_to_idx = {c: i for i, c in enumerate(
            self.vocab)}  # 以char:index的形式返回
        self.idx_to_ch = {i: c for i, c in enumerate(self.vocab)}

    # 类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
    def __getitem__(self, index):
        line = self.lines[index]
        x_str = " "+line  # 初始情况表示零向量，同时确保x=y
        y_str = line+"\n"
        x = torch.zeros([len(x_str), self.vocab_size], dtype=torch.float)
        y = torch.empty(len(x_str), dtype=torch.long)

        y[0] = self.ch_to_idx[y_str[0]]  # 取y的首字符索引

        for i, (x_ch, y_ch) in enumerate(zip(x_str[1:], y_str[1:]), 1):
            x[i][self.ch_to_idx[x_ch]] = 1
            y[i] = self.ch_to_idx[y_ch]

        return x, y  # x是开头为空的字符串，y是字符串加换行符

    def __len__(self):
        return len(self.lines)


trn_ds = DinosDataset()
# 从这里shuffle设置为真的时候，那么DinosDataset中就会随机选取index，获取单词。从而会调用DinoDataset的__getitem__
trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_hh = nn.Linear(hidden_size, hidden_size)  # 做y=ax+b ,aa
        self.linear_hx = nn.Linear(input_size, hidden_size, bias=False)  # ax
        self.linear_output = nn.Linear(hidden_size, output_size)  # y^<t>输出

    def forward(self, h_prev, x):
        h = torch.tanh(self.linear_hh(h_prev) +
                       self.linear_hx(x))  # 一个RNN单元索要输出的a
        y = self.linear_output(h)

        return h, y


model = RNN(trn_ds.vocab_size, hidden_size, trn_ds.vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)  # 随机梯度下降算法


def print_sample(sample_idxs):
    print(trn_ds.idx_to_ch[sample_idxs[0]].upper(), end="")
    [print(trn_ds.idx_to_ch[x], end="") for x in sample_idxs[1:]]


def sample(model):  # 采样
    model.eval()
    # 初始化
    word_size = 0
    newline_idx = trn_ds.ch_to_idx["\n"]
    indices = []
    pred_char_idx = -1

    h_prev = torch.zeros([1, hidden_size], dtype=torch.float)
    x = h_prev.new_zeros([1, trn_ds.vocab_size])

    with torch.no_grad():  # 禁用梯度上下文
        while pred_char_idx != newline_idx and word_size != 50:  # 要么单词最大50，要么找到了换行符
            h_prev, y_pred = model(h_prev, x)
            softmax_scores = torch.softmax(y_pred, dim=1).cpu().numpy().ravel()
            np.random.seed(np.random.randint(1, 5000))
            idx = np.random.choice(
                np.arange(trn_ds.vocab_size), p=softmax_scores)  # 获得概率较大的字符索引
            indices.append(idx)

            x = (y_pred == y_pred.max(1)[0]).float()  # 将概率最大=实际最大的索引置1
            pred_char_idx = idx

            word_size = word_size+1

        if word_size == 50:
            indices.append(newline_idx)  # 如果最大词数是50那就手动添加\n

    return indices


def train_one_epoch(model, loss_fn, optimizer):
    for line_num, (x, y) in enumerate(trn_dl):
        model.train()
        loss = 0
        optimizer.zero_grad()
        h_prev = torch.zeros([1, hidden_size], dtype=torch.float)  # 初始化h_prev

        for i in range(x.shape[1]):  # 一个单词的循环
            h_prev, y_pred = model(h_prev, x[:, i])
            loss = loss+loss_fn(y_pred, y[:, i])
        if (line_num+1) % 100 == 0:  # 读了100个单词
            print_sample(sample(model))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5)  # 裁剪梯度防止梯度爆炸，最大范数是5
        optimizer.step()


def train(model, loss_fn, optimizer, epochs=1):
    for e in range(1, epochs+1):
        print(f'{"-"*20} Epoch {e} {"-"*20}')
        train_one_epoch(model, loss_fn, optimizer)


train(model, loss_fn, optimizer, epochs=5)
