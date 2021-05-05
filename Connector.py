import numpy as np
from model.layers.Layer import Layer
from model.layers.Dense import Dense
import model.math.activations as act
import model.math.optimizer as opt
import model.math.losses as losses
import warnings


class Connector:
    """
    This class used to connect any layers extend from Layer class.
    You could use this class to build a model with many layers.
    """
    def __init__(self, layers):
        self.epoch = None
        self.optimizer = None
        self.loss = None
        self.loss_gradient = None
        self.x = None
        self.y = None
        self.x_batch = []
        self.y_batch = []
        self.layers = layers
        self.iteration = 1
        for i in range(1, len(self.layers)):
            self.layers[i](self.layers[i - 1].units())

    def train(self, x, y, learning_rate=0.01, epoch=100, loss='mse', optimizer='adam', batch_size=32, print_loss=True):
        opt_names, opts = opt.get_optimizers()
        loss_names, loss_function, loss_gradient = losses.get_losses()
        assert loss in loss_names
        if (self.layers[-1].units() > 1) and loss == 'nce':
            warnings.warn("The loss function you selected might be wrong! NCE(Normal Cross Entropy) should be used to "
                          "dichotomy.")
        assert optimizer in opt_names
        self.layers[0](x.shape[0])
        self.loss = loss_function[loss_names.index(loss)]
        self.loss_gradient = loss_gradient[loss_names.index(loss)]
        self.x = x
        self.y = y
        self.epoch = epoch
        self.optimizer = opts[opt_names.index(optimizer)]
        if batch_size > 0:
            for i in range(0, self.x.shape[1] // batch_size + 1):
                self.x_batch.append(self.x[:, i * batch_size:(i + 1) * batch_size])
                self.y_batch.append(self.y[:, i * batch_size:(i + 1) * batch_size])
        else:
            self.x_batch.append(self.x)
            self.y_batch.append(self.y)
        for i in range(1, epoch + 1):
            for batch_index in range(len(self.x_batch)):
                prop_result = self.x_batch[batch_index]
                for layer in self.layers:
                    prop_result = layer.compute_prop(prop_result)
                loss = self.loss(prop_result, self.y_batch[batch_index])
                loss_gra = self.loss_gradient(prop_result, self.y_batch[batch_index])
                cost = np.mean(np.mean(loss, axis=1, keepdims=True), axis=0, keepdims=True)
                gradient = loss_gra
                for layer in reversed(self.layers):
                    dZ, gradient = layer.gradient(gradient)
                    parameters = layer.parameters()
                    layer_inputs = layer.layer_inputs()
                    new_parameters = self.optimizer(learning_rate,
                                                    dZ=dZ, X=layer_inputs,
                                                    W=parameters.get('W'), b=parameters.get('b'),
                                                    VW=parameters.get('VW'), Vb=parameters.get('Vb'),
                                                    SW=parameters.get('SW'), Sb=parameters.get('Sb'),
                                                    iter=self.iteration
                                                    )
                    layer.update(new_parameters)
                print('iter: {}, loss:{}'.format(self.iteration, cost[0, 0]))
                self.iteration += 1

    def test(self, x, y):
        prop_result = x
        for layer in self.layers:
            prop_result = layer.compute_prop(prop_result)
        loss = self.loss(prop_result, y)
        cost = np.sum(np.mean(loss, axis=1, keepdims=True), axis=0, keepdims=True)
        print(prop_result)
        print('test cost:{}, accuracy:{}'.format(cost, 0))


if __name__ == '__main__':
    from Lesson1.dnn_utils_v2 import load_dataset

    x_train_origin, y_train_origin, x_test_origin, y_test_origin, classes = load_dataset()
    x_train = (x_train_origin.reshape(x_train_origin.shape[0], -1) / 255).T
    x_test = (x_test_origin.reshape(x_test_origin.shape[0], -1) / 255).T
    print("数据：")
    print(y_train_origin)
    model = Connector([Dense(10, activation='relu'),
                       Dense(7, activation='relu'),
                       Dense(1, activation='sigmoid')
                       ])
    model.train(x_train, y_train_origin, learning_rate=0.00015, epoch=300, loss='nce', batch_size=0, optimizer='adam')
