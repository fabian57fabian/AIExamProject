import numpy as np
from PerceptronAbstract import PerceptronAbstract
from numpy import dot


class PerceptronSimple(PerceptronAbstract):

    def __init__(self, R):
        super().__init__()
        self.R = R
        self.w = np.zeros(R)
        self.b = 0

    def predict(self, x):
        z = dot(self.w, x) + self.b
        return np.sign(z)

    def get_weights(self):
        return self.w

    def train(self, data, epochs_t=1):
        self.w = np.zeros(self.R)
        self.b = 0
        for e in range(epochs_t):
            n_err = 0
            for x, y in data:
                if y * (dot(self.w, x) + self.b) <= 0:
                    self.w += y * x
                    self.b += y * self.R * self.R
                    n_err += 1
