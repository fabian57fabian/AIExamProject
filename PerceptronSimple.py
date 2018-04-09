import numpy as np
from PerceptronAbstract import PerceptronAbstract
from numpy import dot


class PerceptronSimple(PerceptronAbstract):

    def __init__(self, lr=1):
        super().__init__()
        self.w = np.array(0)
        self.b = np.array(0)
        self.lr = lr
        self.trained = False

    def predict(self, x):
        if not self.trained:
            return 0
        z = dot(self.w, x) + self.b
        return self.sign(z)

    def get_weights(self):
        return self.w

    def train(self, X, Y, epochs_t=1):
        R = len(X[0])
        self.w = np.zeros(R)
        self.b = k = 0
        for _ in range(epochs_t):
            n_err = 0
            for x, y in zip(X, Y):
                if y * (dot(self.w, x) + self.b) <= 0:
                    self.w += y * x
                    self.b += y * R * R
                    k += 1
                    n_err += 1
        self.trained = True

    def sign(self, x):
        if x == 0:
            return 0
        if x > 0:
            return 1
        if x < 0:
            return -1
