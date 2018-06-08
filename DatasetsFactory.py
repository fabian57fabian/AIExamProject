import numpy as np
import csv
from random import randint


class DatasetsFactory:
    @staticmethod
    def printDataset(x, y):
        for r, label in zip(x, y):
            print(r, " ", label)

    @staticmethod
    def data_banknote(path):
        dataset = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            for row in iter_bank:
                y = 1
                if int(row[4]) == 0:
                    y = -1
                x = np.array(row[0:4], dtype=float)
                dataset.append([x, y])
        return dataset

    @staticmethod
    def htru_2(path):
        dataset = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            skip = True
            for row in iter_bank:
                y = 1
                if int(row[8]) == 0:
                    y = -1
                x = np.array(row[0:8], dtype=float)
                dataset.append([x, y])
        return dataset

    @staticmethod
    def simple_points(path):
        dataset = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            next(iter_bank)  # saltare il primo
            for row in iter_bank:
                y = int(row[2])
                x = np.array(row[0:2], dtype=float)
                dataset.append([x, y])
        return dataset

    @staticmethod
    def generate_random_simple_points():
        path = "datasets\\simple_points\\simple_points.data"
        with open(path, "w+") as text_file:
            text_file.write("x, y, label\n")
            for k in range(0, 2500):
                for i in range(2):
                    text_file.write(str(randint(1, 9)) + "," + str(randint(32, 97)) + "," + "1\n")
                for i in range(2):
                    text_file.write(str(randint(32, 97)) + "," + str(randint(1, 9)) + "," + "-1\n")
