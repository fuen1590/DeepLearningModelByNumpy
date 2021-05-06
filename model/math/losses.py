import numpy as np


def get_losses():
    """
    This methods is a interface between activations information and the layers or the model. If
    you add the new functions or delete the old functions, you should update this methods to
    tell the user what activations they can use.

    :return: name_list: The loss function name list.
             methods_norm: The implementations list to propagation.
             methods_gradient: The implementations list to compute gradient of loss.
    """
    name_list = ['mse', 'cce', 'nce']
    methods_norm = [mse, category_cross_entropy, normal_cross_entropy]
    methods_gradient = [mse_gradient, category_cross_entropy_gradient, normal_cross_entropy_gradient]
    return name_list, methods_norm, methods_gradient


def mse(y, label):
    return np.square(y-label)/2


def category_cross_entropy(y, label):
    assert y.shape[0] == label.shape[0] > 1
    return - label * np.log(y)


def normal_cross_entropy(y, label):
    assert y.shape[0] == label.shape[0] == 1
    return -(label*np.log(y)+(1-label)*np.log(1-y))


def mse_gradient(y, label):
    return y-label


def category_cross_entropy_gradient(y, label):
    return -np.divide(label, y)


def normal_cross_entropy_gradient(y, label):
    return -(np.divide(label, y) - np.divide(1-label, 1-y))
