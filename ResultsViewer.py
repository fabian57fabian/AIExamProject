import matplotlib.pyplot as plt
import numpy as np
import csv
from os import listdir

#if new dataset added, please append dataset's info in data array

data = []
data.append({'datasetname': 'simple_separable', 'val_err': 0, 'test_err': 0, 'ephoc': 0, 'tests': {}})
data.append({'datasetname': 'htru_2', 'val_err': 0, 'test_err': 0, 'ephoc': 0, 'tests': {}})
data.append({'datasetname': 'data_banknote', 'val_err': 0, 'test_err': 0, 'ephoc': 0, 'tests': {}})
data.append({'datasetname': 'occupancy', 'val_err': 0, 'test_err': 0, 'ephoc': 0, 'tests': {}})
names = []


def plot_histogram_bests(results):
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
    path = "results/All.txt"
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


def import_tests():
    global data
    path = "results/"
    for _file in listdir(path):
        if _file != "All.txt":
            with open(path + _file, 'r') as file:
                iter_bank = iter(csv.reader(file, delimiter=','))
                header = next(iter_bank)
                print(header)
                next(iter_bank)
                tests = []
                for row in iter_bank:
                    tests.append(np.array(row, dtype=float))
                for d in data:
                    if d['datasetname'] == header[0].strip():
                        attribute = header[1].strip()
                        d['tests'][attribute] = tests
                        break
    return data


def plot_data(data):
    plt.figure(1)
    i = 1
    for d in data:
        plt.subplot(2, 2, i)
        i += 1
        results = [*d['tests']]
        print(results)
        for perc_name in results:
            _data = np.array(d['tests'][perc_name])
            plt.plot(np.array(_data)[:, 0], np.array(_data)[:, 1], 'o-', label=perc_name)
            plt.legend(loc=4)
            plt.ylabel('Val Accurancy')
            plt.xlabel('Ephocs')
            plt.title(d['datasetname'], )
            plt.grid(True)
            plt.gca().set_ylim([0, 110])
    plt.subplots_adjust(left=0.05, right=0.99, hspace=0.30, top=0.95, bottom=0.07)
    plt.show()


def main():
    # results = import_results()
    # plot_histogram_bests(results)
    data = import_tests()
    plot_data(data)


if __name__ == "__main__":
    main()
