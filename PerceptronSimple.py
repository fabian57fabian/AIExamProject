import numpy as np
from PerceptronAbstract import PerceptronAbstract
from numpy import dot


class PerceptronSimple(PerceptronAbstract):

    def __init__(self,  lr=0.01):
        super().__init__()
        self.w = None
        self.b = 0
        self.learning_rate = lr
        self.sum_err = 0

    def predict(self, x):
        z = dot(self.w, x) + self.b
        return np.sign(z)

    def get_weights(self):
        return self.w

    def get_biases(self):
        return self.b

    def train(self, data, epochs_t=1):
        if self.w is None:
            self.w=np.zeros(len(data[0][0]))
        for e in range(epochs_t):
            for x, y in data:
                pred = self.predict(x)
                if y * pred <= 0:
                    error= y -pred
                    self.w += self.learning_rate * error * x
                    self.b += y * self.learning_rate * error
                    self.sum_err += error**2
        return self.sum_err
