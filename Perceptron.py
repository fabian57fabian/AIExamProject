import numpy as np


class Perceptron(object):

    def __init__(self, lr=1):
        self.W = np.array([0])
        self.lr = lr
        self.trained = False

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        if not self.trained:
            return 0
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def get_weights(self):
        return self.W

    def train(self, X, y, epochs_t=1):
        W = np.zeros(len(X[0]) + 1)
        for _ in range(epochs_t):
            for i in range(len(y)):
                x_curr = np.insert(X[i], 0, 1)
                y_new = self.predict(x_curr)
                e = y[i] - y_new
                W = W + self.lr * e * x_curr
        self.W = W
        self.trained = True
