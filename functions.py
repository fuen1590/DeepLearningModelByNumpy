import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(Z, 0)


def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def sigmoid_backward(dA, Z):
    return dA * sigmoid(Z) * (1 - sigmoid(Z))


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh(dA, Z):
    return dA * (1 - tanh(Z) ** 2)
