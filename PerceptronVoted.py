import numpy as np
from PerceptronAbstract import PerceptronAbstract


class PerceptronVoted(PerceptronAbstract):
    v = []
    c = []
    k = 0
    trained = False

    def __init__(self, attr_number):
        super().__init__()
        self.learningRate = 0.1
        self.v = [np.zeros(attr_number)]
        self.c = [0]
        self.attr_number = attr_number

    def predict(self, x_data):
        s = 0
        for c, v in zip(self.c, self.v):
            s += c * np.sign(np.dot(v, x_data))
        return np.sign(s)

    def get_weights(self):
        return self.v

    def train(self, data, epochs_t=1):
        self.v = [np.zeros(self.attr_number)]
        self.c = [0]
        c = 1
        w = np.zeros(self.attr_number)
        for e in range(epochs_t):
            for x, y in data:
                prediction = np.sign(np.dot(w, x))
                if y == prediction:
                    c += 1
                else:
                    w = w + y * x
                    self.v.append(w)
                    self.c.append(c)
                    c = 1
        return self.v, self.c
