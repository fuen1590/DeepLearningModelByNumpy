import numpy as np
from model.math.activations import relu, relu_backward, sigmoid, sigmoid_backward, tanh, tanh_backward
from model.math import optimizer
from model.layers.Layer import Layer

"""
This is a old implementation Dense Layer. Cant be used, just a test.
"""
class Dense(Layer):
    def __init__(self, unit_num, last_unit_num, activation="relu", optimizer="adam"):
        """
        This class is a implementation of dense layer
        :param unit_num: You shuold know this mean.
        :param last_unit_num: This parameter will determine the params shape.
        :param activation: The activation function, you should use these:["relu", "tanh", "sigmoid"]
        :param optimizer: This layer's optimizer.
        """
        self.W = 2 * np.random.randn(unit_num, last_unit_num) / np.sqrt(last_unit_num)
        self.b = np.zeros((unit_num, 1))
        self.dW = 0
        self.db = 0
        self.Z = 0
        self.A = 0
        self.dZ = 0
        self.dA = 0
        self.input = 0
        self.activation = activation
        self.optimizer = optimizer
        # momentum parameters:
        self.VW = 0
        self.Vb = 0
        # RMSprop parameters:
        self.SW = 0
        self.Sb = 0
        # all of above parameters are Adam parameters, and this:
        self.iter = 1

    def forward(self, input):
        self.input = input
        self.Z = np.dot(self.W, self.input) + self.b
        if self.activation == "relu":
            self.A = relu(self.Z)
        elif self.activation == "sigmoid":
            self.A = sigmoid(self.Z)
        elif self.activation == "tanh":
            self.A = tanh(self.Z)
        return self.A

    def backward(self, dA, learning_rate):
        if self.activation == "relu":
            self.dZ = relu_backward(dA, self.Z)
        elif self.activation == "sigmoid":
            self.dZ = sigmoid_backward(dA, self.Z)
        elif self.activation == "tanh":
            self.dZ = tanh_backward(dA, self.Z)

        """
        There are old gradient decent method, and now they are still available.
        But I dont recommend you to use these code.
        """
        # self.dW = np.dot(self.dZ, self.input.T) / self.input.shape[1]
        # self.db = np.sum(self.dZ, keepdims=True) / self.input.shape[1]
        # last_dA = np.dot(self.W.T, self.dZ)
        # self.W -= learning_rate * self.dW
        # self.b -= learning_rate * self.db

        last_dA = np.dot(self.W.T, self.dZ)
        if self.optimizer == 'gd':
            self.W, self.b = optimizer.gradient_decent(self.W, self.b, self.dZ, self.input, learning_rate)
        elif self.optimizer == 'momentum':
            self.W, self.b, self.VW, self.Vb = optimizer.momentum(self.W, self.b,
                                                                  self.VW, self.Vb,
                                                                  self.dZ, self.input,
                                                                  learning_rate)
        elif self.optimizer == 'RMS':
            self.W, self.b, self.SW, self.Sb = optimizer.RMSprop(self.W, self.b,
                                                                 self.SW, self.Sb,
                                                                 self.dZ, self.input,
                                                                 learning_rate)
        elif self.optimizer == 'adam':
            self.W, self.b, self.VW, self.Vb, self.SW, self.Sb, self.iter = \
                optimizer.adam(self.W, self.b,
                               self.VW, self.Vb,
                               self.SW, self.Sb,
                               self.dZ, self.input,
                               self.iter, learning_rate)
        return last_dA

    def show_params(self):
        print(self.W)
