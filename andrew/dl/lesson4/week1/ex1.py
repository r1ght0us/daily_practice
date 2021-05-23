# 卷积神经网络的实现

import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams["image.interpolation"] = "nearest"  # 按图像比例缩小的效果设置
plt.rcParams["image.cmap"] = "gray"

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))

# 作业大纲
'''
卷积函数，包括：
    - 零填充
    - 卷积窗口
    - 正向卷积
    - 反向卷积（可选）
池化函数，包括：
    - 正向池化
    - 创建mask
    - 分配值
    - 反向池化（可选）
'''

# 对于每个正向函数，都有其对应的反向等式。因此，在正向传播模块的每一步中，都将一些参数存储在缓存中。这些参数用于在反向传播时计算梯度。

# 在这一部分，你将构建卷积层的每一步。首先实现两个辅助函数：一个用于零填充，另一个用于计算卷积函数本身。

# 零填充


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   "constant", constant_values=0)  # 第一个是多少个图片m，第四个是通道数量
    #

    return X_pad


'''
# 测试padding0填充是否功能正常
x=rs.randn(4,3,3,2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])
plt.show()
'''

# 卷积的单个步骤

# 图像的每一部分与过滤器进行wx+b，然后相加的值返回到已经完成过滤的相应部分中


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)+b

    Z = np.sum(s)

    return Z

# 卷积神经网络--正向传递


def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = w.shape  # 获取相应矩阵的大小

    stride = hparameters["stride"]  # 超参数
    pad = hparameters["pad"]

    n_H = 1+int((n_H_prev+2*pad-f)/stride)
    n_W = 1+int((n_W_prev+2*pad-f)/stride)

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        A_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f  # 取过滤器所要求的大小

                    # 取各个通道的固定大小
                    a_slice_prev = a_prev_pad[vert_start:vert_end,
                                              horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(np.multiply(
                        a_slice_prev, W[:, :, :, c])+b[:, :, :, c])  # 求出z

    cache = (A_prev, W, b, hparameters)  # 保存缓存后面反向传播可以用

    return Z, cache


# 池化层

def pool_forward(A_prev, hparameters, mode="max"):
    '''
    正向池化，没有用于反向传播的参数
    '''
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1+(n_H_prev-f)/stride)  # 进行池化的时候不需要padding
    n_W = int(1+(n_W_prev-f)/stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f

                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    return A, cache

'''
# 测试函数
A_prev = rs.randn(2, 4, 4, 3)
hparameters = {"stride": 1, "f": 4}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode="average")
print("mode = average")
print("A =", A)
'''