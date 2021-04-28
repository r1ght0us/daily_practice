# 初始化

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))

train_X, train_Y, test_X, test_Y = load_dataset()

# 神经网络模型
'''
零初始化 ：在输入参数中设置initialization = "zeros"。
随机初始化 ：在输入参数中设置initialization = "random"，这会将权重初始化为较大的随机值。
He初始化 ：在输入参数中设置initialization = "he"，这会根据He等人（2015）的论文将权重初始化为按比例缩放的随机值
'''


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    grads = {}
    costs = []
    m = X.shape[1]
    layer_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    else:
        parameters = initialize_parameters_he(layer_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = compute_loss(a3, Y)

        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print(i, cost)
            costs.append(cost)
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("learning rate="+str(learning_rate))
    plt.show()

    return parameters

# 零初始化


def initialize_parameters_zeros(layer_dims):
    parameters = {}

    L = len(layer_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# 随机初始化


def initialize_parameters_random(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = rs.randn(layer_dims[l], layer_dims[l-1])*10
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# parameters = model(train_X, train_Y, initialization = "random")
# predictions_train = predict(train_X, train_Y, parameters)
# predictions_test = predict(test_X, test_Y, parameters)


def initialize_parameters_he(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = rs.randn(layer_dims[l],
                                          layer_dims[l-1])*np.sqrt(2./layer_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


parameters = model(train_X, train_Y, initialization="he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

# 决策边界
#plt.title("Model with He initialization")
# axes = plt.gca() #获得当前画布的轴信息
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
#plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

'''
1. 不同的初始化会导致不同的结果
2. 随机初始化用于打破对称性，并确保不同的隐藏单元可以学习不同的东西
3. 不要初始化为太大的值
4. 初始化对于带有ReLU激活的网络非常有效。
'''