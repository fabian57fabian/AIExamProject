import numpy as np
import matplotlib.image as mpimg
import os
import csv


class DatasetsFactory:

    @staticmethod
    def diseasedTrees(path):
        x = []
        y = []
        first = True
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if not first:
                    if row[0] == 'n':
                        y.append(1)
                    else:
                        y.append(-1)
                    row.pop(0)
                    row = np.array(row, dtype=float)
                    x.append(row)
                else:
                    first = False
        return x, y

    @staticmethod
    def simplePoints():
        x = np.array([
            [1, 90], [2, 45], [3, 88], [2, 16], [1, 8], [1, 9],
            [1, 12], [1, 19], [2, 24], [2, 19], [3, 50], [2, 15], [3, 8],
            [18, 2], [40, 3], [19, 2], [30, 3], [26, 2], [29, 3],
            [38, 4], [59, 1], [12, 1], [95, 4], [32, 2], [27, 1], [3, 9]
        ], dtype=float)
        y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        return x, y

    @staticmethod
    def getIris(path):
        x = []
        y = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if row[4] == 'Iris-setosa':
                    y.append(1)
                else:
                    y.append(-1)
                row.pop(4)
                row = np.array(row, dtype=float)
                x.append(row)
        return x, y

    @staticmethod
    def get_cmc(path):
        x = []
        y = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if row[9] == '1':
                    y.append(1)
                else:
                    y.append(-1)
                row.pop(9)
                row = np.array(row, dtype=float)
                x.append(row)
        return x, y

    @staticmethod
    def get_shuttle(path):
        x = []
        y = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                if row[9] == '4' or row[9] == '1':
                    if row[9] == '1':
                        y.append(1)
                    else:
                        y.append(-1)
                    row.pop(9)
                    row.pop(0)
                    row = np.array(row, dtype=float)
                    x.append(row)
        return x, y

    @staticmethod
    def getMovement_AAL(pathFolder, testing):
        x = []
        y = []
        x1 = []
        y1 = []
        sequences = []
        labels = []
        path_target = pathFolder + "\\MovementAAL_target.csv"
        with open(path_target, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            first = True
            for row in reader:
                if not first:
                    sequences.append(row[0])
                    labels.append(row[1])
                else:
                    first = False
        to_training = False
        for i in range(len(sequences)):
            if i >= 289:
                to_training = True
            path_target = pathFolder + "\\MovementAAL_RSS_" + sequences[i] + ".csv"
            with open(path_target, 'r') as file:
                reader = csv.reader(file, delimiter=',')
                first = True
                for row in reader:
                    if not first:
                        if not to_training:
                            y.append(int(labels[i]))
                            row = np.array(row, dtype=float)
                            x.append(row)
                        else:
                            y1.append(int(labels[i]))
                            row = np.array(row, dtype=float)
                            x1.append(row)
                    else:
                        first = False
        return x, y, x1, y1
