import numpy as np
import matplotlib.image as mpimg
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

    def realisticDataset(path1, path2):
        x = []
        y = []
        x, y = DatasetsFactory.getImagesFromPath(path1, x, y, 1)
        x, y = DatasetsFactory.getImagesFromPath(path2, x, y, -1)
        return x, y

    def getImagesFromPath(path, x, y, y_label):
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.png'):
                    img = DatasetsFactory.getImageAsArray(path + "\\" + filename)
                    x.append(img)
                    y.append(y_label)
        print(x[0])
        return x, y

    def getImageAsArray(path):
        #TODO: load png as 1d array
        return mpimg.imread(path)
