# 神经风格迁移（NST）

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from nst_utils import *
import pdb

torch.set_printoptions(linewidth=200)  # 设置输出选项，linewidth:每行200个字符
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')  # 加载gpu

# 神经迁移
model = VGG19_StyleTransfer()

# 计算内容代价


def compute_content_cost(a_C, a_G):
    m, H, W, C = a_G.shape

    a_C_unrolled = a_C.view(C, -1)  # 展开维度变成(C,H*W)
    a_G_unrolled = a_G.view(C, -1)  # 维度相同用哪个都行

    J_content = torch.mean((a_C_unrolled-a_G_unrolled)**2)/(4*H*W*C)

    return J_content

# 计算单层的风格损失


def compute_layer_style_cost(a_S, a_G):
    N, C, H, W = a_G.shape
    a_S_unrolled = a_S.view(C, -1)
    a_G_unrolled = a_G.view(C, -1)

    GS = torch.mm(a_S_unrolled, a_S_unrolled.t())
    GG = torch.mm(a_G_unrolled, a_G_unrolled.t())

    J_style_layer = torch.sum((GS-GG)**2)/(4*(C**2)*((H*W)**2))

    return J_style_layer


# 风格权重
style_layers = [0, 5, 10, 19, 28]
style_layer_coeffs = [0.2, 0.2, 0.2, 0.2, 0.2]

# 从几个不同的层次“合并”风格损失


def compute_style_cost(style_layers, style_layer_coeffs, generated_im_results, style_im_results):

    J_style = 0

    for layer_num, coeff in zip(style_layers, style_layer_coeffs):
        a_S = style_im_results[layer_num]
        a_G = generated_im_results[layer_num]
        J_style_layer = compute_layer_style_cost(a_S, a_G)  # 计算当前层的style_cost

        # 当前层的（style_cost x coeff）添加到整体风格损失（J_style）中
        J_style = J_style+(coeff*J_style_layer)

    return J_style


def get_costs(model, content_im_results, style_im_results, style_layers, style_layer_coeffs, content_layer, generated_image):
    generated_im_results = model(generated_image)

    a_C = content_im_results[content_layer]
    a_G = generated_im_results[content_layer]
    J_content = compute_content_cost(a_C, a_G)
    J_style = compute_style_cost(
        style_layers, style_layer_coeffs, generated_im_results, style_im_results)

    return J_content, J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha*J_content + beta*J_style

    return J

# 解决优化问题


# 加载内容图片
content_image = imageio.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)
# 加载风格图片
style_image = imageio.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)


# 生成随机噪点图像，这将有助于生成的图像的内容更快速地匹配“内容”图像的内容。
generated_image = generate_noise_image(content_image)

content_layer = 21
model = VGG19_StyleTransfer(layers=style_layers + [content_layer])  # 加载VGG模型
content_image = torch.tensor(content_image).float().to(device)
style_image = torch.tensor(style_image).float().to(device)
generated_image = torch.tensor(generated_image).float().to(device)
model = model.to(device)

optimizer = optim.Adam([generated_image.requires_grad_()], lr=2.0)
content_im_results = model(content_image)
style_im_results = model(style_image)


for layer in content_im_results.keys():
    # 返回tensor的复制
    content_im_results[layer] = content_im_results[layer].detach()
    style_im_results[layer] = style_im_results[layer].detach()

params = {
    'model': model,
    'style_layers': style_layers,
    'style_layer_coeffs': style_layer_coeffs,
    'content_layer': content_layer,
    'generated_image': generated_image,
    'content_im_results': content_im_results,
    'style_im_results': style_im_results
}


def train(model, style_layers, style_layer_coeffs, content_layer, generated_image,
          content_im_results, style_im_results, num_iterations=200):

    for i in range(num_iterations):
        # Print every 20 iteration.
        Jc, Js = get_costs(model, content_im_results, style_im_results,
                           style_layers, style_layer_coeffs, content_layer, generated_image)
        optimizer.zero_grad()
        Jt = total_cost(Jc, Js)
        Jt.backward()
        optimizer.step()
        if i % 20 == 0:
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt.item()))
            print("content cost = " + str(Jc.item()))
            print("style cost = " + str(Js.item()))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)


def save_image(path, image):
    # Un-normalize the image so that it looks good
    image = image.clone()
    # image = image.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # detach共用内存所以会发生bug，所以上面的代码transpose调换维之后后面跑的style直接指数上涨
    image = image.detach().cpu().numpy().transpose(0, 2, 3, 1)
    image += CONFIG.MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')

    imageio.imsave(path, image)


train(**params, num_iterations=200)
