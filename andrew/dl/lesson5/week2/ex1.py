# 使用单词向量表示来构建Emojifier表情符号

import numpy as np
from torch.nn.init import xavier_normal_, xavier_uniform_
from models.emo_utils import *
import emoji
import matplotlib.pyplot as plt
import torch as pt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

pt.set_printoptions(linewidth=200)
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

X_train, Y_train = read_csv("data/train_emoji.csv")
X_test, Y_test = read_csv("data/tesss.csv")

maxLen = len(max(X_train, key=len).split())

# 测试数据集是否正常
# index=1
# print(X_train[index],label_to_emoji(Y_train[index]))

Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)  # 将Y的维度从(m,1)转换成(m,5)

#  实现Emojifier-V1


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    "data/glove.6B.50d.txt")
'''
word_to_index：字典将单词映射到词汇表中的索引（400,001个单词，有效索引范围是0到400,000）
index_to_word：字典从索引到词汇表中对应词的映射
word_to_vec_map：将单词映射到其GloVe向量表示的字典。
'''

# word = "cucumber"
# index = 289846
# print("the index of", word, "in the vocabulary is", word_to_index[word])
# print("the", str(index) + "th word in the vocabulary is", index_to_word[index])


def sentence_to_avg(sentence, word_to_vec_map):  # 计算句子的一个平均值
    words = (sentence.lower()).split()  # 分隔字符串
    avg = pt.zeros(50, dtype=pt.float32)

    for w in words:
        avg += pt.tensor(word_to_vec_map[w], dtype=pt.float32)
    avg = avg/len(words)

    return avg


class Emo_Dataset(Dataset):
    def __init__(self, X, Y, word_to_vec_map):
        self.word_to_vec_map = word_to_vec_map
        self.X = X
        self.Y = Y
        super().__init__()

    def __getitem__(self, index):  # 所有Dataset的子类都要重写__getitem__以及__len__
        x = sentence_to_avg(self.X[index], word_to_vec_map)
        y = self.Y[index]
        return x, y

    def __len__(self):  # 提供数据集的大小
        return self.X.shape[0]


trn_ds = Emo_Dataset(X_train, Y_train, word_to_vec_map)
trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True)
test_ds = Emo_Dataset(X_test, Y_test, word_to_vec_map)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

model = nn.Linear(in_features=50, out_features=5)
nn.init.xavier_uniform_(model.weight)  # 权重初始化
loss_fn = nn.CrossEntropyLoss()  # 将LogSoftMax和NLLLoss集成到一个类中
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 实现随机梯度下降算法


def compute_accuracy(model, dl):
    model.eval()
    num_examples = len(dl)  # 因为分批就是1所以len就是其长度
    correct_preds = 0
    with pt.no_grad():
        for x, y in dl:
            z = model(x)
            pred_cls = pt.softmax(z, dim=-1).argmax()  # arg返回最大值的索引
            # item张量的值作为标准 Python 数字返回。这仅适用于具有一个元素的张量。
            correct_preds += (y == pred_cls).item()
    model.train()  # 切换为训练模式
    return correct_preds/num_examples


def train(model, loss_fn, optimizer, trn_dl, num_epochs=400):
    n_y = 5  # y的类别
    n_h = 50  # Glove的向量大小
    model.train()
    for e in range(num_epochs):
        for x, y in trn_dl:
            z = model(x)
            optimizer.zero_grad()
            loss = loss_fn(z, y.long())
            loss.backward()
            optimizer.step()
        if e % 100 == 0:
            print(f'Epoch: {e} --- cost = {loss}')
            accuracy = compute_accuracy(model, trn_dl)
            print(f"Accuracy: {accuracy}")


train(model, loss_fn, optimizer, trn_dl)

print("Training set accuracy:", compute_accuracy(model, trn_dl))
print("Test set accuracy:", compute_accuracy(model, test_dl))


def test_custom_sentences(model, sentences, labels, word_to_vec_map):
    model.eval()
    num_examples = len(sentences)
    correct_preds = 0
    y = pt.tensor(labels).to(device)
    with pt.no_grad():
        for (sentence, label) in zip(sentences, y):
            x = sentence_to_avg(sentence, word_to_vec_map).to(device)
            z = model(x)
            pred_cls = pt.softmax(z, dim=-1).argmax()
            correct_preds += (label == pred_cls).item()
            print(sentence, label_to_emoji(pred_cls.item()))

    print('\nAccuracy:', correct_preds/num_examples)


X_my_sentences = ["i adore you", "i love you", "funny lol",
                  "lets play with a ball", "food is ready", "not feeling happy"]
Y_my_labels = [0, 0, 2, 1, 4, 3]

test_custom_sentences(model, X_my_sentences, Y_my_labels, word_to_vec_map)

