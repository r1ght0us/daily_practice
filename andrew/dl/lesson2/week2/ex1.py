# 优化算法

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1)))


def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters)//2

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - \
            learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - \
            learning_rate*grads["db"+str(l+1)]

    return parameters

# mini-batch梯度下降


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]
    mini_batches = []

    permutation = list(rs.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m/mini_batch_size)  # math.floor向下取整

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,
                                  num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,
                                  num_complete_minibatches*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# 动量梯度下降

def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(L):
        v["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)

    return v


# 如果β等于0那么就成标准的梯度下降
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters)//2

    for l in range(L):
        v["dW"+str(l+1)] = beta*v["dW"+str(l+1)]+(1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta*v["db"+str(l+1)]+(1-beta)*grads["db"+str(l+1)]

        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - \
            learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - \
            learning_rate*v["db"+str(l+1)]
    return parameters, v


# adam算法

def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW"+str(l+1)] = beta1*v["dW"+str(l+1)] + \
            (1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + \
            (1-beta1)*grads["db"+str(l+1)]

        v_corrected["dW"+str(l+1)] = v["dW"+str(l+1)]/(1-(beta1**t))
        v_corrected["db"+str(l+1)] = v["db"+str(l+1)]/(1-(beta1**t))

        s["dW"+str(1+l)] = beta2*s["dW"+str(1+l)] + \
            (1-beta2)*(grads["dW"+str(1+l)]**2)
        s["db"+str(1+l)] = beta2*s["db"+str(1+l)] + \
            (1-beta2)*(grads["db"+str(1+l)]**2)

        s_corrected["dW"+str(l+1)] = s["dW"+str(l+1)]/(1-(beta2)**t)
        s_corrected["db"+str(l+1)] = s["db"+str(l+1)]/(1-(beta2)**t)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)]-learning_rate*(
            v_corrected["dW" + str(l + 1)]/np.sqrt(s_corrected["dW" + str(l + 1)]+epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)]-learning_rate*(
            v_corrected["db" + str(l + 1)]/np.sqrt(s_corrected["db" + str(l + 1)]+epsilon))
    return parameters, v, s

# 不同优化算法的模型


train_X, train_Y = load_dataset()


def model(X, Y, layer_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    L = len(layer_dims)
    costs = []
    t = 0

    parameters = initialize_parameters(layer_dims)

    if optimizer == "gd":  # 初始优化器
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        minibatches = random_mini_batches(
            X, Y, mini_batch_size)  # 重新划分num_epochs次mini_batch

        for minibatch in minibatches:  # 在mini_batch中进行操作
            (minibatch_X, minibatch_Y) = minibatch

            a3, caches = forward_propagation(minibatch_X, parameters)

            cost = compute_cost(a3, minibatch_Y)

            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(
                    parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters,v = update_parameters_with_momentum(
                    parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t+1  # t最小从1开始
                parameters,v,s = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        if print_cost and i % 1000 == 0:
            print(i, cost)
        elif print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("epchos(per 100)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


layers_dims = [train_X.shape[0], 5, 2, 1]
#小批量梯度下降


# parameters = model(train_X, train_Y, layers_dims, optimizer="gd")

# # Predict
# predictions = predict(train_X, train_Y, parameters)

# # Plot decision boundary
# plt.title("Model with Gradient Descent optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# plot_decision_boundary(lambda x: predict_dec(
#     parameters, x.T), train_X, train_Y)


#动量小批量下降
# parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# # Predict
# predictions = predict(train_X, train_Y, parameters)

# # Plot decision boundary
# plt.title("Model with Momentum optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#adam批量梯度下降
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)