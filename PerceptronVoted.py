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
        self.R = R

    def predict(self, x_data):
        s = 0
        for c, v in zip(self.c, self.v):
            s += c * np.sign(np.dot(v, x_data))
        return np.sign(s)

    def get_weights(self):
        return self.v

    def train(self, data, epochs_t=1):
        y_label = 1
        self.v = [np.zeros(self.R)]
        self.c = [0]
        for e in range(epochs_t):
            for x, y in data:
                s = np.dot(self.v[-1], x)
                if y_label == np.sign(s):
                    self.c[-1] += 1
                else:
                    self.v.append(self.v[-1] + y * x)
                    self.c.append(1)
        return self.v, self.c
