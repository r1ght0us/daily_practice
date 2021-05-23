# 人脸识别

import cv2
from numpy.core.numeric import identity
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from inception_model import *

# 将人脸图像编码为128维向量
FRmodel = faceRecoModel()

# 计算tripletloss计算损失


class TripletLoss(nn.Module):
    def __init__(self, alpha=0.2):  # α是损失函数中的边距
        self.alpha = alpha
        super(TripletLoss, self).__init__()

    def forward(self, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = torch.sum((anchor-positive)**2, dim=-1)
        neg_dist = torch.sum((anchor-negative)**2, dim=-1)
        basic_loss = pos_dist-neg_dist+self.alpha
        # torch.new_zeros返回一个size用填充的大小的张量0
        loss = torch.sum(torch.max(basic_loss, basic_loss.new_zeros(1)))

        return loss


# 加载已经训练好的模型
loss_fn = TripletLoss()
optimizer = optim.Adam(FRmodel.parameters())
load_weights_from_FaceNet(FRmodel)


def img_to_encoding(image_path, model):
    model.eval()
    img1 = cv2.imread(image_path, 1)  # 1>0加载三通道图像
    img = img1[..., ::-1]  # 翻转,实现RGB到BGR通道的转换
    img = np.around(np.transpose(img, (2, 0, 1))/255.0,
                    decimals=12)  # np.around()四舍五入到给定的小数数

    img_tensor = torch.tensor(np.array([img]), dtype=torch.float)

    out = FRmodel(img_tensor)
    # tensor.detach()返回一个新的Tensor,从当前的计算图中分离出来,返回的Tensor和原Tensor共享相同的存储空间，但是返回的 Tensor 永远不会需要梯度
    # cpu()返回此对象在CPU内存中的副本。
    return out.detach().cpu().numpy()


database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


# 该函数检查前门摄像头拍摄到的图片（image_path）是否是本人
def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)

    dist = np.linalg.norm(encoding-database[identity])  # 求范数

    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

#verify("images/camera_3.jpg", "kian", database, FRmodel)


# 从人脸验证到人脸识别
def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)

    min_dist = 100  # 初始化最小距离
    for (name, db_enc) in database.items(): # items() 函数以列表返回可遍历的(键, 值) 元组数组
        dist = np.linalg.norm(db_enc-encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

#who_is_it("images/camera_0.jpg", database, FRmodel)

