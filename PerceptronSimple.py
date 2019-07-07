import numpy as np
from PerceptronAbstract import PerceptronAbstract
from numpy import dot


class PerceptronSimple(PerceptronAbstract):

    def __init__(self, R):
        super().__init__()
        self.R = R
        self.w = np.zeros(R)
        self.b = 0
        self.n_err = 0

    def predict(self, x):
        z = dot(self.w, x) + self.b
        return np.sign(z)

    def get_weights(self):
        return self.w

    def get_biases(self):
        return self.b

    def train(self, data, epochs_t=1):
        for e in range(epochs_t):
            for x, y in data:
                if y * (dot(self.w, x) + self.b) <= 0:
                    self.w += y * x
                    self.b += y * self.R * self.R
                    self.n_err += 1
        return self.n_err