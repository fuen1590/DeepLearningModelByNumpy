import numpy as np
from functions import relu, relu_backward, sigmoid, sigmoid_backward


class Layer:
    def __init__(self, unit_num, last_unit_num, activation="relu"):
        # 注意初始化问题，要根据前一层的单元数量进行初始化，前一层数量越多，初始化参数越小
        self.W = np.random.randn(unit_num, last_unit_num) / np.sqrt(last_unit_num)
        self.b = np.zeros((unit_num, 1))
        self.dW = 0
        self.db = 0
        self.Z = 0
        self.A = 0
        self.dZ = 0
        self.dA = 0
        self.input = 0
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.Z = np.dot(self.W, self.input) + self.b
        if self.activation == "relu":
            self.A = relu(self.Z)
        elif self.activation == "sigmoid":
            self.A = sigmoid(self.Z)
        return self.A

    def backward(self, dA, learning_rate):
        if self.activation == "relu":
            self.dZ = relu_backward(dA, self.Z)
        elif self.activation == "sigmoid":
            self.dZ = sigmoid_backward(dA, self.Z)
        self.dW = np.dot(self.dZ, self.input.T) / self.input.shape[1]
        self.db = np.sum(self.dZ, keepdims=True) / self.input.shape[1]
        last_dA = np.dot(self.W.T, self.dZ)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return last_dA

    def show_params(self):
        print(self.W)
