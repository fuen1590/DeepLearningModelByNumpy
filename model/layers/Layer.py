import numpy as np
import model.math.activations as act
import model.math.optimizer


class Layer:
    def __init__(self, units: int, activation: str):
        """
        This class is the root class of all the other layers in the future.
        You could extend this layer to implement your own compute function.
        :param units: You should know this params.
        :param activation: The activation function, you should use these:["relu", "tanh", "sigmoid"]
        """
        act_names, acts, acts_gradient = act.get_activations()
        assert activation in act_names
        self._units = units
        self.activation = acts[act_names.index(activation)]
        self._gradient = acts_gradient[act_names.index(activation)]
        self._achieve_init = False
        self.X = None
        self.W = 0
        self.b = 0
        self.Z = 0
        self.A = 0
        self.dW = 0
        self.db = 0
        self.dZ = 0
        self.dA = 0

    def __call__(self):
        if not self._achieve_init:
            self._achieve_init = True

    def compute_prop(self, inputs):
        if not self._achieve_init:
            raise Exception("You must call layer(input_shape) methods to initiate the layer.")
        self.X = inputs

    def gradient(self, dA):
        self.dA = dA
        self.dZ = self._gradient(dA, self.Z)
        dA_1 = np.matmul(self.W.T, self.dZ)
        return self.dZ, dA_1

    def parameters(self):
        return {'W': self.W, 'b': self.b}

    def layer_inputs(self):
        return self.X

    def units(self):
        return self._units

