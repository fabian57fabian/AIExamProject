import numpy as np
from PerceptronAbstract import PerceptronAbstract


class PerceptronVotedWithBias(PerceptronAbstract):

    def __init__(self, R, learningRate=0.5):
        super().__init__()
        self.learningRate = learningRate
        self.w = np.zeros(R)
        self.b = 1
        self.v = []
        self.c = []
        self.bias = []

    def set(self):
        self.v = []
        self.c = []
        self.bias = []

    def train(self, data, ephocs_t=1):
        c = 1
        for i in range(ephocs_t):
            for x, y in data:
                predicted = np.sign(np.dot(self.w, x) + self.b)
                if predicted != y:
                    self.w = self.w + (y * x) * self.learningRate
                    self.v.append(self.w)
                    self.c.append(c)
                    self.bias.append(self.b)
                    c = 1
                    self.b = self.b + y
                else:
                    c = c + 1
        return self.w, self.b

    def predict(self, x):
        p = 0
        i = 0
        while i < len(self.v):
            prediction = np.sign(np.dot(x, self.v[i]) + self.bias[i])
            p += self.c[i] * prediction
            i = i + 1
        return np.sign(p)

    def get_weights(self):
        return self.v