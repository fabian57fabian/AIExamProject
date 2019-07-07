from PerceptronVoted import PerceptronVoted
from PerceptronSimple import PerceptronSimple
from DatasetsFactory import DatasetsFactory
from ResultsViewer import main as plotAll
import time
import random
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--ephocs", type=int, default=500,
                    help="Number of ephocs to train (default: 500)")
parser.add_argument("--train-number", type=int, default=500,
                    help="Number of examples for training (default: 500)")
parser.add_argument("--validation-number", type=int, default=500,
                    help="Number of examples for Validation (default: 500)")
parser.add_argument("--test-number", type=int, default=1000,
                    help="Number of examples for testing (default: 1000)")
args = parser.parse_args()
train_n = args.train_number
ephocs = args.ephocs
validation_n = args.validation_number
test_n = args.test_number

# DEFAULT SETTINGS vars
dataPath = "datasets"
random_key = 4

accuracies = []

# Prceptrons
perceptron_types = []
perceptron_types.append({'number': 1, 'name': 'Perceptron', 'class': PerceptronSimple})
perceptron_types.append({'number': 2, 'name': 'PerceptronVoted', 'class': PerceptronVoted})


def load_datasets_info():
    datasets = []
    datasets.append({'name': 'simple_separable',
                     'data': DatasetsFactory.simple_points,
                     'data_path': '/simple_points/simple_points.data'})
    datasets.append({'name': 'htru_2',
                     'data': DatasetsFactory.htru_2,
                     'data_path': '/htru_2/HTRU_2.arff'})
    datasets.append({'name': 'data_banknote',
                     'data': DatasetsFactory.data_banknote,
                     'data_path': '/banknote_authentication/data_banknote_authentication.txt'})
    datasets.append({'name': 'occupancy',
                     'data': DatasetsFactory.occupancy,
                     'data_path': '/Occupancy/datatraining.txt'})
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
    path = "results/All.txt"
    with open(path, "w+") as text_file:
        text_file.write("Name, Type, ephoc, [Validation accuracy, Testing accuracy]")
        for acc in accuracies:
            text_file.write("\n" + acc[0] + ',' + acc[1] + ',' + str(acc[2]) + ',' + str(acc[3]) + ',' + str(acc[4]))


def train_dataset(dataset, type):
    tests = []
    print("\nTraining " + dataset['name'] + " with " + type['name'])
    data = dataset['data'](dataPath + dataset['data_path'])
    R = len(data[0][0])
    random.Random(random_key).shuffle(data)
    train_data = data[0:train_n]
    validation_data = data[train_n: train_n + validation_n]
    test_data = data[train_n + validation_n: train_n + validation_n + test_n]
    my_perceptron = type['class'](R)  # create perceptron
    accuracy_validation = 0
    print("\nTime | Epoch | Accurancy on Validation | Time spent")
    for epoch in range(ephocs):
        start = time.time()
        errs = my_perceptron.train(train_data, 1)  # train perceptron
        accuracy_validation = test_with(my_perceptron, validation_data)  # validate perceptron
        elapsed = time.time() - start
        test = [epoch + 1, accuracy_validation, elapsed]
        tests.append(test)
        # if (epoch+1) % delay_epoch_print == 0:
        print("{} | E {:03} | V {:.2f} | D {:03.2f}".format(str(datetime.datetime.now()), test[0], test[1], test[2]))
        if errs == 0:
            break
    accuracy_test = test_with(my_perceptron, test_data)
    save_results(tests, dataset, len(train_data), len(validation_data), type['name'])
    global accuracies
    accuracies.append([dataset['name'], type['name'], ephocs, accuracy_validation, accuracy_test])


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


def save_results(tests, dataset, train_n, validation_n, name):
    path = "results/" + name + ' ' + dataset['name'] + ".data"
    with open(path, "w+") as text_file:
        text_file.write(
            dataset['name'] + "," + name + ",[Train=" + str(train_n) + ",Validation=" + str(
                validation_n) + "]")
        text_file.write("\nEphocs, Accurancy val, Time")
        for test in tests:
            text_file.write("\n" + str(test).replace("[", "").replace("]", ""))


if __name__ == "__main__":
    main()
