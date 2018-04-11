import matplotlib.pyplot as plt
import numpy as np
import csv
from os import listdir

data = []
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


def import_tests():
    data = []
    data.append({'datasetname': 'simple_separable', 'PV_tests': [], 'P_tests': []})
    data.append({'datasetname': 'diseased_trees', 'PV_tests': [], 'P_tests': []})
    data.append({'datasetname': 'htru_2', 'PV_tests': [], 'P_tests': []})
    data.append({'datasetname': 'data_banknote', 'PV_tests': [], 'P_tests': []})
    data.append({'datasetname': 'data_occupancy', 'PV_tests': [], 'P_tests': []})
    path = "results\\"
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
                        attribute = 'P_tests' if header[1].strip() == 'PerceptronSimple' else 'PV_tests'
                        d[attribute] = tests
                        break
    return data


def plot_data(data):
    plt.figure(1)
    i = 1
    for d in data:
        plt.subplot(2, 3, i)
        i += 1
        PV_data = np.array(d['PV_tests'])
        P_data = np.array(d['P_tests'])
        plt.plot(np.array(PV_data)[:, 0], np.array(PV_data)[:, 1], 'o-', label="Perceptron Voted")
        plt.plot(np.array(P_data)[:, 0], np.array(P_data)[:, 1], 'o-', label="Perceptron")
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
