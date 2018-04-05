import numpy as np
import csv
from random import randint


class DatasetsFactory:

    @staticmethod
    def createAttributesDictionary():
        array = []
        array.append({"age": 'numeric_value'})
        array.append({"admin.": 1, "blue-collar": 2, "entrepreneur": 3, "housemaid": 4, "management": 5, "retired": 6,
                      "self-employed": 7, "services": 8, "student": 9, "technician": 10, "unemployed": 11,
                      "unknown": 0})
        array.append({"divorced": 1, "married": 2, "single": 3, "unknown": 0})
        array.append(
            {"basic.4y": 1, "basic.6y": 2, "basic.9y": 3, "high.school": 4, "illiterate": 5, "professional.course": 6,
             "university.degree": 7, "unknown": 8})
        array.append({"no": 1, "yes": 2, "unknown": 0})
        array.append({"no": 1, "yes": 2, "unknown": 0})
        array.append({"no": 1, "yes": 2, "unknown": 0})
        array.append({"cellular": 1, "telephone": 2})
        array.append(
            {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
             "nov": 11, "dec": 12})
        array.append({"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7})
        array.append({"duration": 'numeric_value'})
        array.append({"campaign": 'numeric_value'})
        array.append({"pdays": 'numeric_value'})
        array.append({"failure": 1, "nonexistent": 2, "success": 3})
        array.append({"emp.var.rate": 'numeric_value'})
        array.append({"cons.price.idx": 'numeric_value'})
        array.append({"cons.conf.idx": 'numeric_value'})
        array.append({"euribor3m": 'numeric_value'})
        array.append({"nr.employed": 'numeric_value'})
        array.append({"y": 'numeric_value'})
        return array

    @staticmethod
    def bankMarketing(path, max_examples_1, max_examples_N1):
        attributes_dict = DatasetsFactory.createAttributesDictionary()
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=';'))
            next(iter_bank)
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0: break
                curr_y = 0
                if row[20] == 'no':
                    if max_examples_N1 <= 0:
                        skip = True
                    else:
                        curr_y = -1
                        # y.append(-1)
                        skip = False
                    max_examples_N1 -= 1
                else:
                    if max_examples_1 <= 0:
                        skip = True
                    else:
                        curr_y = 1
                        # y.append(1)
                        skip = False
                    max_examples_1 -= 1
                if not skip:
                    x_row = []  # create an array with age
                    x_row.append(int(row[0]))
                    for i in range(1, 10):  # attributes 1-9 have strings values
                        x_row.append(attributes_dict[i][row[i]])
                    x_row.append(int(row[13]))
                    x_row.append(attributes_dict[13][row[14]])
                    for i in range(15, 20):
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

    @staticmethod
    def eye_state(path, max_examples_1, max_examples_N1):
        x = []
        y = []
        with open(path, 'r') as file:
            iter_bank = iter(csv.reader(file, delimiter=','))
            skip = True
            for row in iter_bank:
                if max_examples_1 < 0 and max_examples_N1 < 0:
                    break
                curr_y = 0
                if int(row[14]) == 0:
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
                    for i in range(0, 14):
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

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
                    for i in range(2, 7):#skip date and number
                        x_row.append(float(row[i]))
                    x.append(x_row)
                    y.append(curr_y)
        return x, y

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
