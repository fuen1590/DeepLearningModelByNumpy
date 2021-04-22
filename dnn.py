import numpy as np
from layer import Layer

np.set_printoptions(suppress=True)
class Dnn:
    def __init__(self, x, y, units, learning_rate, iters, loss="across"):
        assert (len(units) >= 2)
        self.X = x
        self.Y = y
        self.learning_rate = learning_rate
        self.iters = iters
        self.layers = []
        self.layers.append(Layer(units[0], x.shape[0], activation="relu"))
        for i in range(1, len(units) - 1):
            self.layers.append(Layer(units[i], units[i - 1]))
        self.layers.append(Layer(units[-1], units[-2], activation="sigmoid"))
        self.loss = loss
        self.cost = []

    def loss_function(self, y):
        return -(self.Y*np.log(y)+(1-self.Y)*np.log(1-y))

    def loss_function_grid(self, y):
        return -(np.divide(self.Y, y) - np.divide(1-self.Y, 1-y))

    def fit(self):
        last_output = 0
        last_grid = 0
        for i in range(self.iters):
            for j in range(len(self.layers)):
                if j == 0:
                    last_output = self.layers[j].forward(self.X)
                else:
                    last_output = self.layers[j].forward(last_output)
            loss = self.loss_function(last_output)
            loss_grid = self.loss_function_grid(last_output)
            cost = np.sum(loss, axis=1, keepdims=True)/self.X.shape[1]
            self.cost.append(cost)
            print("iter %d loss: %f"%(i, cost))
            for k in range(-1, -len(self.layers)-1, -1):
                if k == -1:
                    last_grid = self.layers[k].backward(loss_grid, self.learning_rate)
                else:
                    last_grid = self.layers[k].backward(last_grid, self.learning_rate)

                # self.layers[k].grid_decent(self.learning_rate)


    def test_model(self, input, labels):
        last_output = 0
        for j in range(len(self.layers)):
            if j == 0:
                last_output = self.layers[j].forward(input)
            else:
                last_output = self.layers[j].forward(last_output)
        output = np.ones(last_output.shape)
        output[last_output<0.5] = 0
        accuracy = 1 - np.sum(labels - output)/labels.shape[1]
        return output, accuracy

    def structure(self):
        for layer in self.layers:
            print(layer.W.shape)
            print(layer.b.shape)
            layer.show_params()

if __name__ == "__main__":
    from Lesson1.dnn_utils_v2 import load_dataset
    x_train_origin, y_train_origin, x_test_origin, y_test_origin, classes = load_dataset()
    x_train = (x_train_origin.reshape(x_train_origin.shape[0], -1)/255).T
    x_test = (x_test_origin.reshape(x_test_origin.shape[0], -1)/255).T
    print("数据：")
    print(y_train_origin)
    # 学习率过大，导致模型损失固定在0.643974，
    model = Dnn(x_train, y_train_origin, [20, 7, 5, 1], 0.0055, 1000)
    model.fit()
    output, accuracy = model.test_model(x_test, y_test_origin)
    print(output)
    print(y_test_origin)
    print("准确率：%f"%(accuracy))
    import matplotlib.pyplot as plt

    plt.plot(np.squeeze(model.cost))
    plt.xlabel("iterrations")
    plt.ylabel("cost")
    plt.show()
