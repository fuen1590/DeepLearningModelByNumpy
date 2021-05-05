import numpy as np
from model.math.activations import relu, relu_backward, sigmoid, sigmoid_backward, tanh, tanh_backward
from model.math import optimizer
from model.layers.Layer import Layer


class Dense(Layer):
    def __init__(self, units, activation="relu"):
        """
        This class is a implementation of dense layer
        :param units: You should know this params.
        :param activation: The activation function, you should use these:["relu", "tanh", "sigmoid"]
        """
        super(Dense, self).__init__(units=units, activation=activation)
        # momentum parameters:
        self.VW = 0
        self.Vb = 0
        # RMSprop parameters:
        self.SW = 0
        self.Sb = 0

    def __call__(self, inputs_shape):
        """
        This methods used to achieve the init of the layer.
        The initiation of the parameter use HE methods:
            w = random(0~1)/np.sqrt(last_layer_nums)
        :param inputs_shape: A number of the last layer's units num.
        """
        assert not self._achieve_init
        self.W = 2 * np.random.randn(self._units, inputs_shape) / np.sqrt(inputs_shape)
        self.b = np.zeros((self._units, 1))
        super(Dense, self).__call__()

    def compute_prop(self, inputs):
        super(Dense, self).compute_prop(inputs)
        self.Z = np.matmul(self.W, self.X) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def parameters(self):
        params = super(Dense, self).parameters()
        return {'W': params.get('W'), 'b': params.get('b'), 'VW': self.VW, 'Vb': self.Vb, 'SW': self.SW, 'Sb': self.Sb}

    def update(self, new_parameters):
        self.W = new_parameters.get('new_W')
        self.b = new_parameters.get('new_b')
        self.VW = new_parameters.get('new_VW')
        self.Vb = new_parameters.get('new_Vb')
        self.SW = new_parameters.get('new_SW')
        self.Sb = new_parameters.get('new_Sb')

