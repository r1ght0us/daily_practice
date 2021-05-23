from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
# Image_data_format()，返回默认图像的维度顺序(“channels_first"或"channels_last”)
# 选择channels_first：返回(3,256,256)
# 选择channels_last：返回(256,256,3)


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T


def model(input_shape):
    X_input = Input(input_shape)  # 将带有shape的张量，作为一张输入图片
    X = ZeroPadding2D((3, 3))(X_input)  # 为宽和高进行3*3的零填充

    X = Conv2D(32, (7, 7), strides=(1, 1), name="conv0")(
        X)  # 32个过滤器，核大小(7*7)，步长为1
    X = BatchNormalization(axis=3, name="bn0")(X)  # 在通道上进行归一化
    X = Activation("relu")(X)

    X = MaxPooling2D((2, 2), name="max_pool")(X)

    X = Flatten()(X)
    # Dense 实现以下操作： output = activation(dot(input, kernel) + bias),就是你常用的的全连接层
    X = Dense(1, activation="sigmoid", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


def HappyModel(input_shape):
    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(1, 1))(X_input)
    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 2), padding="valid")(X)  # pool_size表示池化层大小

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Flatten()(X)
    Y = Dense(1, activation="sigmoid")(X) #units=1,输出空间维度。

    model = Model(inputs=X_input, outputs=Y, name='HappyModel')

    return model


happyModel = HappyModel((64, 64, 3))  # 创建模型
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-08, decay=0.0), loss="binary_crossentropy", metrics=['accuracy'])
# decay学习率衰减
# metrics 模型在训练和测试期间要评估的指标的形式
happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)
