import numpy as np

"""
This file implements the activation functions.
You can see the dA parameter in _backward functions. Because use dA will compute the 
    dZ more convenient. 
"""


def get_activations():
    """
    This methods is a interface between activations information and the layers or the model. If
    you add the new functions or delete the old functions, you should update this methods to
    tell the user what activations they can use.

    :return: name_list: The activation name list.
             methods_norm: The implementations list to propagation.
             methods_gradient: The implementations list to compute gradient of Z
    """
    name_list = ['relu', 'leak_relu', 'tanh', 'sigmoid', 'softmax']
    methods_norm = [relu, leak_relu, tanh, sigmoid, softmax]
    methods_gradient = [relu_backward, leak_relu_backward, tanh_backward, sigmoid_backward, softmax_backward]
    return name_list, methods_norm, methods_gradient


"""Normal function."""


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z), axis=0)


def relu(Z):
    return np.maximum(Z, 0)


def leak_relu(Z):
    return np.maximum(Z, 0.01*Z)


def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


"""Gradient function."""


def softmax_backward(dA, Z):
    return dA * softmax(Z) * (1 - softmax(Z))


def sigmoid_backward(dA, Z):
    return dA * sigmoid(Z) * (1 - sigmoid(Z))


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def leak_relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0.01
    return dZ


def tanh_backward(dA, Z):
    return dA * (1 - tanh(Z) ** 2)
