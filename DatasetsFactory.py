import numpy as np
import csv
from random import randint


class DatasetsFactory:
    @staticmethod
    def printDataset(x, y):
        for r, label in zip(x, y):
            print(r, " ", label)

    @staticmethod
    def data_banknote(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if int(row[4]) == 0:
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    x_row = []  # create an array with age
                    for i in range(0, 4):
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

    @staticmethod
    def cmc(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if int(row[9]) == 2:
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                elif int(row[9]) == 1:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                else:
                    skip = True
                if not skip:
                    x_row = []  # create an array with age
                    for i in range(0, 9):
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

    @staticmethod
    def iris(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if row[4] == 'Iris-setosa':
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    x_row = []  # create an array with age
                    for i in range(0, 4):
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

    @staticmethod
    def htru_2(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if int(row[8]) == 0:
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    x_row = []  # create an array with age
                    for i in range(0, 8):
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

    @staticmethod
    def data_occupancy(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            next(iter_bank)  # saltare il primo
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if int(row[7]) == 0:
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    x_row = []  # create an array with age
                    for i in range(2, 7):  # skip date and number
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

    @staticmethod
    def simple_points(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            next(iter_bank)  # saltare il primo
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if int(row[2]) == -1:
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    row.pop(2)
                    x.append(np.array(row, dtype=float))
                    y.append(curr_y)
        return x, y

    @staticmethod
    def diseasedTrees(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        first = True
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            next(iter_bank)  # skip first row
            curr_y = 0
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if row[0] == 'n':
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    row.pop(0)
                    x_row = np.array(row, dtype=float)
                    x.append(x_row)
                    y.append(curr_y)
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

    @staticmethod
    def generate_random_simple_points():
        path = "datasets\\simple_points\\simple_points.data"
        with open(path, "w+") as text_file:
            text_file.write("x, y, label\n")
            for i in range(0, 5000):
                text_file.write(str(randint(1, 9)) + "," + str(randint(32, 97)) + "," + "1\n")
            for i in range(0, 5000):
                text_file.write(str(randint(32, 97)) + "," + str(randint(1, 9)) + "," + "-1\n")