from PerceptronVoted import PerceptronVoted
from PerceptronSimple import PerceptronSimple
from DatasetsFactory import DatasetsFactory
import datetime
# https://plot.ly/matplotlib/bar-charts/
import matplotlib.pyplot as plt
import time
import numpy as np
import csv

data = []
names = []


def plot_histogram(results):
    plt.ylabel("datasets")
    plt.xlabel("results")
    plt.title("Datasets accurancy")

    y_pos = np.arange(len(data))
    plt.barh(y_pos, data, align='center', alpha=0.5)
    plt.yticks(y_pos, names)

    plt.show()


def import_results():
    global data
    global names
    results = []
    names = []
    path = "results\\All.txt"
    with open(path, 'r') as file:
        iter_bank = iter(csv.reader(file, delimiter=','))
        # skip first two rows
        next(iter_bank)
        next(iter_bank)
        for row in iter_bank:
            name = row[1]
            if row[0] == 'PerceptronSimple':
                name = "P " + name
            else:
                name = "PV " + name
            names.append(name)
            data.append(float(row[5]))
            result = {'type': row[0], 'name': row[1], 'ephoc': int(row[2]), 'test_len': int(row[3]),
                      'train_len': int(row[4]), 'accurancy': float(row[5]), 'time': float(row[6])}
            results.append(result)
    return np.array(results)


def main():
    results = import_results()
    plot_histogram(results)


if __name__ == "__main__":
    main()
