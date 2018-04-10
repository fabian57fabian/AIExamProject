from PerceptronVoted import PerceptronVoted
from PerceptronSimple import PerceptronSimple
from DatasetsFactory import DatasetsFactory
import matplotlib.pyplot as plt
import time
import random
import numpy as np

# SETTINGS vars
dataPath = "datasets"
to_plot = True
plot_n = 331
use_voted = False
epochs = [1, 3, 4, 7, 10, 14, 18, 20, 25, 30, 50, 70, 90, 100]
train_n = 500
validation_n = 500
accuracies = []


def main():
    if to_plot:
        plt.figure(1)
    global use_voted
    datasets = load_datasets_info()
    sets = []
    sets.append(datasets[0])
    for dataset in datasets:
        use_voted = False
        train_dataset(dataset)
        use_voted = True
        train_dataset(dataset)
    save_accurancies_testing()
    plot_data(accuracies)


def save_accurancies_testing():
    path = "results\\All.txt"
    with open(path, "w+") as text_file:
        text_file.write("Name, Type, ephoc, Validation accuracy, Testing accuracy]")
        for acc in accuracies:
            text_file.write("\n" + acc[0] + ',' + acc[1] + ',' + str(acc[2]) + ',' + str(acc[3]) + ',' + str(acc[4]))


def train_dataset(dataset):
    tests = []
    print("\nTraining " + dataset['name'] + " with " + get_perceptron_type_str())
    data = dataset['data'](dataPath + dataset['data_path'])
    R = len(data[0][0])
    random.Random(4).shuffle(data)
    train_data = data[0:train_n]
    validation_data = data[train_n: train_n + validation_n]
    best_a = 0
    best_e = 0
    best_perc = 0
    for epoch in epochs:
        my_perceptron = get_perceptron(R)  # create perceptron
        start = time.time()
        my_perceptron.train(train_data, epoch)  # train perceptron
        accuracy_validation = test_with(my_perceptron, validation_data)  # validate perceptron
        elapsed = time.time() - start
        tests.append([epoch, accuracy_validation, elapsed])
        print(tests[-1])
        if best_a < accuracy_validation:
            best_a = accuracy_validation
            best_e = epoch
            best_perc = my_perceptron
    test_data = data[train_n * 2:max(len(data), 2000)]
    accuracy_test = test_with(best_perc, test_data)
    subplot(tests, dataset['name'], get_perceptron_type_str())
    save_results(tests, dataset, accuracy_test, train_n, validation_n, len(test_data))
    global accuracies
    accuracies.append([dataset['name'], get_perceptron_type_str(), best_e, best_a, accuracy_test])


def subplot(tests, name, perc_type):
    if to_plot:
        global plot_n
        plt.subplot(plot_n)
        plot_n += 1
        plt.plot(np.array(tests, dtype=int)[:, 0], np.array(tests)[:, 1], 'o-', label=perc_type + ' ' + name)
        plt.ylabel('Accurancy')
        plt.xlabel('Ephocs')
        plt.title(name)
        plt.grid(True)
        plt.gca().set_ylim([0, 110])


def get_perceptron_type_str():
    if use_voted:
        return 'PerceptronVoted'
    else:
        return 'PerceptronSimple'


def get_perceptron(R):
    if use_voted:
        return PerceptronVoted(R)
    else:
        return PerceptronSimple(R)


def test_with(my_perceptron, data):
    err = 0
    i = len(data)
    for d, p in data:
        result = my_perceptron.predict(d)
        if abs(result - p) != 0:
            err += 1
        i -= 1
    accurancy = 100 - (err / len(data) * 100)
    return accurancy


def plot_data(accuracies):
    if to_plot:
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()


def save_results(tests, dataset, accurancy_test, train_n, validation_n, test_n):
    path = "results\\" + get_perceptron_type_str() + dataset['name'] + ".data"
    with open(path, "w+") as text_file:
        text_file.write(
            dataset['name'] + ", " + get_perceptron_type_str() + " [Train=" + str(train_n) + ", Validation=" + str(
                validation_n) + "]")
        text_file.write("\nEphocs, Accurancy, Time")
        for test in tests:
            text_file.write("\n" + str(test).replace("[", "").replace("]", ""))


def load_datasets_info():
    datasets = []
    datasets.append({'name': 'simple_separable',
                     'data': DatasetsFactory.simple_points,
                     'data_path': '\\simple_points\\simple_points.data'})
    datasets.append({'name': 'diseased_trees',
                     'data': DatasetsFactory.diseasedTrees,
                     'data_path': '\\wilt\\testing.csv'})
    datasets.append({'name': 'htru_2',
                     'data': DatasetsFactory.htru_2,
                     'data_path': '\\htru_2\\HTRU_2.arff'})
    datasets.append({'name': 'data_banknote',
                     'data': DatasetsFactory.data_banknote,
                     'data_path': '\\banknote_authentication\\data_banknote_authentication.txt'})
    datasets.append({'name': 'data_occupancy',
                     'data': DatasetsFactory.data_occupancy,
                     'data_path': '\\occupancy_data\\datatest.txt'})
    return datasets


if __name__ == "__main__":
    main()
