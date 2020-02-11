import numpy as np
from PerceptronAbstract import PerceptronAbstract
import random


class PerceptronVoted(PerceptronAbstract):

    def __init__(self, learningRate=0.1):
        super().__init__()
        self.learningRate = learningRate
        self.w = None
        self.b = 1
        self.v = []
        self.c = []
        self.tmp_c = 1
        self.bias = []
        self.n_err = 1

    def get_biases(self):
        return self.b

    def get_votes(self):
        return self.v

    def train(self, data, ephocs_t=1):
        random.Random().shuffle(data)
        c = self.tmp_c
        w = self.w = np.zeros(len(data[0][0])) if self.w is None else self.w
        b = self.b
        for i in range(ephocs_t):
            self.n_err = 0
            for x, y in data:
                predicted = np.sign(np.dot(w, x) + b)
                if predicted != y:
                    w = w + (y * x) * self.learningRate
                    self.v.append(w)
                    self.c.append(c)
                    self.bias.append(self.b)
                    c = 1
                    b += y
                    self.n_err += 1
                else:
                    c += 1
        self.w = w
        self.tmp_c = c
        self.b = b
        return self.n_err

    def predict(self, x):
        p = 0
        i = 0
        while i < len(self.v):
            prediction = np.sign(np.dot(x, self.v[i]) + self.bias[i])
            p += self.c[i] * prediction
            i = i + 1
        return np.sign(p)

    def get_weights(self):
        return self.w
