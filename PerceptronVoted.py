import numpy as np
from PerceptronAbstract import PerceptronAbstract


class PerceptronVoted(PerceptronAbstract):
    v = []
    c = []
    k = 0
    trained = False

    def __init__(self, R):
        super().__init__()
        self.v = [np.zeros(R)]
        self.c = [0]
        self.k = 0
        self.R = R

    def predict(self, x_data):
        s = 0
        for i in range(self.k + 1):
            s += self.c[i] * np.sign(np.dot(self.v[i], x_data))
        return np.sign(s)

    def get_weights(self):
        return self.v

    def train(self, data, epochs_t=1):
        y_label = 1
        self.v = [np.zeros(self.R)]
        self.c = [0]
        self.k = 0
        for e in range(epochs_t):
            for x, y in data:
                s = np.dot(self.v[self.k], x)
                if y_label == np.sign(s):
                    self.c[self.k] += 1
                else:
                    self.v.append(self.v[self.k] + y * x)
                    self.c.append(1)
                    self.k += 1
        return self.v, self.c
