import numpy as np
import os


class DatasetsFactory:
    def simplePoints():
        x = np.array([
            [1, 90], [2, 45], [3, 88], [2, 16], [1, 8], [1, 9],
            [1, 12], [1, 19], [2, 24], [2, 19], [3, 50], [2, 15],
            [18, 2], [40, 3], [19, 2], [30, 3], [26, 2], [29, 3],
            [38, 4], [59, 1], [12, 1], [95, 4], [32, 2], [27, 1],
        ], dtype=float)
        y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
                     dtype=float)
        return x, y

    def realisticDataset(path):

        return [[1], [-1]], [1, -1]
