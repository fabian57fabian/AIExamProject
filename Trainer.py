from PerceptronVoted import PerceptronVoted as PerceptronVoted2
from PerceptronVotedWithBias import PerceptronVotedWithBias as PerceptronVoted
from PerceptronSimple import PerceptronSimple
from DatasetsFactory import DatasetsFactory
from ResultsViewer import main as plotAll
import time
import random
import numpy as np

# SETTINGS vars
dataPath = "datasets"
epochs = [20]
train_n = 500
test_n = 500
accuracies = []

perceptron_types = []
perceptron_types.append({'number': 1, 'name': 'Perceptron', 'class': PerceptronSimple})
perceptron_types.append({'number': 3, 'name': 'PerceptronVoted', 'class': PerceptronVoted})


def load_datasets_info():
    datasets = []
    datasets.append({'name': 'simple_separable',
                     'data': DatasetsFactory.simple_points,
                     'data_path': '\\simple_points\\simple_points.data'})
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


def printTime(elapsedTOTAL):
    m, s = divmod(elapsedTOTAL, 60)
    h, m = divmod(m, 60)
    print("Total time: ")
    print("%d:%02d:%02d" % (h, m, s))


def main():
    startTOTAL = time.time()
    datasets = load_datasets_info()
    sets = []
    sets.append(datasets[0])
    for dataset in datasets:
        for type in perceptron_types:
            train_dataset(dataset, type)
    save_accurancies_testing()
    elapsedTOTAL = time.time() - startTOTAL
    printTime(elapsedTOTAL)
    plotAll()
    return 0


def save_accurancies_testing():
    path = "results\\All.txt"
    with open(path, "w+") as text_file:
        text_file.write("Name, Type, ephoc, [Validation accuracy, Testing accuracy]")
        for acc in accuracies:
            text_file.write("\n" + acc[0] + ',' + acc[1] + ',' + str(acc[2]) + ',' + str(acc[3]) + ',' + str(acc[4]))


def train_dataset(dataset, type):
    tests = []
    print("\nTraining " + dataset['name'] + " with " + type['name'])
    data = dataset['data'](dataPath + dataset['data_path'])
    R = len(data[0][0])
    random.Random(4).shuffle(data)
    train_data = data[0:train_n]
    validation_data = data[train_n: train_n + round(test_n / 2)]
    test_data = data[train_n + round(test_n / 2): train_n + test_n]
    best_a = 0
    best_e = 0
    best_perc = 0
    for epoch in epochs:
        my_perceptron = type['class'](R)  # create perceptron
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
    accuracy_test = test_with(best_perc, test_data)
    save_results(tests, dataset, len(train_data), len(validation_data), len(test_data), type['name'])
    global accuracies
    accuracies.append([dataset['name'], type['name'], best_e, best_a, accuracy_test])


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


def save_results(tests, dataset, train_n, validation_n, test_n, name):
    path = "results\\" + name + ' ' + dataset['name'] + ".data"
    with open(path, "w+") as text_file:
        text_file.write(
            dataset['name'] + "," + name + ",[Train=" + str(train_n) + ",Validation=" + str(
                validation_n) + "]")
        text_file.write("\nEphocs, Accurancy val, Time")
        for test in tests:
            text_file.write("\n" + str(test).replace("[", "").replace("]", ""))


if __name__ == "__main__":
    main()
