import numpy as np
from PerceptronAbstract import PerceptronAbstract


class PerceptronVoted(PerceptronAbstract):
    v = []
    c = []
    k = 0
    trained = False

    def __init__(self):
        super().__init__()
        self.v = []
        self.c = []
        self.trained = False

    def train(self, x, y, epochs_t=1):
        y_label = 1
        k = 0
        c = []
        v = [np.zeros(len(x[0]))]
        yNew = 0
        for e in range(epochs_t):
            c.append(0)
            for i in range(len(x)):
                result = np.dot(v[k], x[i])
                yNew = self.sign(result)
                if yNew == y_label:
                    c[k] += 1
                v.append(v[k] + np.dot(y[i], x[i]))
                c.append(1)
                k += 1
        self.trained = True
        self.c = c
        self.v = v
        self.k = k
        return v, c

    def predict(self, x_data):
        if not self.trained:
            return 0
        s = 0
        for i in range(self.k):
            s += self.c[self.k] * np.sign(np.dot(self.v[i], x_data))
        return self.sign(s)

    def sign(self, x):
        if x == 0:
            return 0
        if x > 0:
            return 1
        if x < 0:
            return -1
