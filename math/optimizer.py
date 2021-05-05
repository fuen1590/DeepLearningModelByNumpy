import numpy as np


def get_optimizers():
    names = ['gd', 'momentum', 'RMS', 'adam']
    optimizers = [gradient_decent, momentum, RMSprop, adam]
    return names, optimizers


def gradient_decent(learning_rate, **kwargs):
    """
    This is a normal gradient decent optimizer.
    If you use mini-batch size equals 1, this will be a Stochastic Gradient Decent(SGD).
    So the detail will not be implemented in here, you could adjust the mini-batch size to
        implement the different gradient_decent.

    :param learning_rate: The step size of the length when you update the parameters.
    :param kwargs: It should include the parameters W and b you need to update, in addition, please
                   give the gradient of Z computed by the reverse activation function, it will be
                   used for the optimizer to calculate W and b bias derivatives, the specific
                   calculation process can be derived from reverse propagation, here only for
                   implementation.
    """
    W = kwargs.get('W')
    b = kwargs.get('b')
    dZ = kwargs.get('dZ')
    X = kwargs.get('X')
    dW = np.dot(dZ, X.T) / X.shape[1]
    db = np.sum(dZ, keepdims=True) / X.shape[1]
    W -= learning_rate * dW
    b -= learning_rate * db
    return {'new_W': W, 'new_b': b}


def momentum(learning_rate, **kwargs):
    """
    Momentum optimizer.

    :param learning_rate: The step size of the length when you update the parameters.
    :param kwargs: It should include..（英语实在不想写了）这里应该包含你所需要更新的W和b，除此之外，请把通过
                   反向激活函数得到的Z的导数值传回，用于优化器计算W和b的偏导数，
                   还应包含W和b的指数加权平均VW和Vb，用于momentum计算一阶矩，具体的计算流程可以从反向传播中推导，
                   这里仅作实现.
    """
    W = kwargs.get('W')
    b = kwargs.get('b')
    VW = kwargs.get('VW')
    Vb = kwargs.get('Vb')
    dZ = kwargs.get('dZ')
    X = kwargs.get('X')
    moment = 0.9
    dW = np.dot(dZ, X.T) / X.shape[1]
    db = np.sum(dZ, keepdims=True) / X.shape[1]
    VW = moment * VW + (1 - moment) * dW
    Vb = moment * Vb + (1 - moment) * db
    W -= learning_rate * VW
    b -= learning_rate * Vb
    return {'new_W': W, 'new_b': b, 'new_VW': VW, 'new_Vb': Vb}


def RMSprop(learning_rate, **kwargs):
    """
    RMS optimizer.

    :param learning_rate: The step size of the length when you update the parameters.
    :param kwargs: It should include..（英语实在不想写了）这里应该包含你所需要更新的W和b，除此之外，请把通过
                   反向激活函数得到的Z的导数值传回，用于优化器计算W和b的偏导数，
                   还应包含W和b的平方指数加权平均SW和Sb(请忽略这个憨批一样的变量名)，用于momentum计算一阶矩，
                   具体的计算流程可以从反向传播中推导，这里仅作实现.
    """
    W = kwargs.get('W')
    b = kwargs.get('b')
    SW = kwargs.get('SW')
    Sb = kwargs.get('Sb')
    dZ = kwargs.get('dZ')
    X = kwargs.get('X')
    weight = 0.9
    epsilon = 0.00000001
    dW = np.dot(dZ, X.T) / X.shape[1]
    db = np.sum(dZ, keepdims=True) / X.shape[1]
    SW = weight * SW + (1 - weight) * (dW ** 2)
    Sb = weight * Sb + (1 - weight) * (db ** 2)
    W -= learning_rate * (dW / (np.sqrt(SW) + epsilon))
    b -= learning_rate * (db / (np.sqrt(Sb) + epsilon))
    return {'new_W': W, 'new_b': b, 'new_SW': SW, 'new_Sb': Sb}


def adam(learning_rate, **kwargs):
    """
    Adaptive moment estimation optimizer.
    Adam is a complex method combined RMS and momentum.

    :param learning_rate: The step size of the length when you update the parameters.
    :param kwargs: It should include..（英语实在不想写了）这里应该包含你所需要更新的W和b，除此之外，请把通过
                   反向激活函数得到的Z的导数值传回，用于优化器计算W和b的偏导数，
                   还应包含W和b的平方指数加权平均SW和Sb(请忽略这个憨批一样的变量名)和普通指数加权平均VW,Vb
                   用于momentum计算一阶矩，另外，请计算每层作用adam的次数iter，用于对指数加权平均做误差修正！
                   具体的计算流程可以从反向传播中推导，这里仅作实现.
    """
    W = kwargs.get('W')
    assert W is not None
    b = kwargs.get('b')
    VW = kwargs.get('VW')
    Vb = kwargs.get('Vb')
    SW = kwargs.get('SW')
    Sb = kwargs.get('Sb')
    dZ = kwargs.get('dZ')
    X = kwargs.get('X')
    iteration = kwargs.get('iter')
    moment = 0.9
    weight = 0.999
    epsilon = 1e-8
    dW = np.dot(dZ, X.T) / X.shape[1]
    db = np.sum(dZ, keepdims=True) / X.shape[1]
    VW = moment * VW + (1 - moment) * dW
    Vb = moment * Vb + (1 - moment) * db
    SW = weight * SW + (1 - weight) * (dW ** 2)
    Sb = weight * Sb + (1 - weight) * (db ** 2)
    # Fixing the loss
    VcW = VW / (1 - moment ** iteration)
    Vcb = Vb / (1 - moment ** iteration)
    ScW = SW / (1 - weight ** iteration)
    Scb = Sb / (1 - weight ** iteration)
    W -= learning_rate * (VcW / (np.sqrt(ScW) + epsilon))
    b -= learning_rate * (Vcb / (np.sqrt(Scb) + epsilon))
    return {'new_W': W, 'new_b': b, 'new_VW': VW, 'new_Vb': Vb, 'new_SW': SW, 'new_Sb': Sb}
