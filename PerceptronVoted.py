import numpy as np


class PerceptronVoted(object):
    v = []
    c = []
    k = 0
    trained = False

    def __init__(self):
        self.v = []
        self.c = []
        self.trained = False

    def train(self, x, y, epochs_t=1):
        y_label = 1
        k = 0
        c = []
        v = [np.zeros(len(x[0]))]
        for e in range(epochs_t):
            c.append(0)
            for i in range(len(x)):
                yNew = self.sign(np.dot(v[k], x[i]))
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

    """
    def dot(self, x_arr, y_arr):
        sum = 0
        for i in range(len(x_arr)):
            if not isinstance(x_arr[i], float) and not isinstance(x_arr[i], int):
                break;
            sum = sum + (x_arr[i] * y_arr[i])
        return sum
    """

    def sign(self, x):
        if x == 0:
            return 0
        if x > 1:
            return 1
        if x < 1:
            return -1
