import numpy as np


class Perceptron1(object):

    def __init__(self, lr=1):
        self.W = 0
        self.lr = lr
        self.trained = False

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        if not self.trained:
            return 0
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def get_weights(self):
        return self.W

    def train(self, X, y, epochs_t=0.1):
        self.W = np.zeros(len(X[0]))
        print(len(X[0]))
        for _ in range(epochs_t):
            for i in range(len(y)):
                x = X[i]
                y_new = self.predict(x)
                e = y[i] - y_new
                b = self.lr * e * x
                if b != []:
                    self.W += b
        print(self.W)
        self.trained = True
