from PerceptronVoted import PerceptronVoted
from Perceptron import Perceptron
from DatasetsFactory import DatasetsFactory
import datetime

dataPath = "datasets"


def abs(number):
    if number == 0:
        return number
    if number < 0:
        return number * -1
    return number


def testData(x, y, ephocs, to_predict, labels):
    print("Train: ", len(x), "Test: ", len(to_predict), "Ephocs: ", ephocs)
    myVoted = Perceptron()
    myVoted.train(x, y, ephocs)
    err = 0
    i = len(to_predict)
    for d, p in zip(to_predict, labels):
        result = myVoted.predict(d)
        if abs(result - p) != 0:
            err += 1
        i -= 1
    errors = (err / len(to_predict) * 100)
    print("Accurancy: ", 100 - errors, "%", "Errors: ", errors, "%")
    return 100 - errors


def simple_separable(ephocs, train, test):
    path = dataPath + "\\simple_points"
    to_predict, labels = DatasetsFactory.simple_points(path + "\\simple_points.data", test / 2, test / 2)
    x, y = DatasetsFactory.simple_points(path + "\\simple_points.data", train / 2, train / 2)
    print("len: ", len(x), " ", len(to_predict))
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


def diseased_trees(ephocs, train, test):
    path = dataPath + "\\wilt"
    to_predict, labels = DatasetsFactory.diseasedTrees(path + "\\testing.csv", test / 2, test / 2)
    x, y = DatasetsFactory.diseasedTrees(path + "\\training.csv", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


def data_banknote(ephocs, train, test):
    basePath = dataPath + "\\banknote_authentication"
    to_predict, labels = DatasetsFactory.data_banknote(basePath + "\\data_banknote_authentication.txt", test / 2,
                                                       test / 2)
    x, y = DatasetsFactory.data_banknote(basePath + "\\data_banknote_authentication.txt", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


def data_occupancy(ephocs, train, test):
    basePath = dataPath + "\\occupancy_data"
    to_predict, labels = DatasetsFactory.data_occupancy(basePath + "\\datatest.txt", test / 2, test / 2)
    x, y = DatasetsFactory.data_occupancy(basePath + "\\datatraining.txt", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


def htru_2(ephocs, train, test):
    basePath = dataPath + "\\htru_2"
    to_predict, labels = DatasetsFactory.htru_2(basePath + "\\HTRU_2.arff", test / 2, test / 2)
    x, y = DatasetsFactory.htru_2(basePath + "\\HTRU_2.arff", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


def iris(ephocs, train, test):
    basePath = dataPath + "\\iris"
    to_predict, labels = DatasetsFactory.iris(basePath + "\\iris.data", test / 2, test / 2)
    x, y = DatasetsFactory.iris(basePath + "\\iris.data", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


def cmc(ephocs, train, test):
    basePath = dataPath + "\\cmc"
    to_predict, labels = DatasetsFactory.cmc(basePath + "\\cmc.data", test / 2, test / 2)
    x, y = DatasetsFactory.cmc(basePath + "\\cmc.data", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(x), len(to_predict)


tests = []
datasets = []
datasets.append(["simple_separable", simple_separable])
# datasets.append(["diseased_trees", diseased_trees])
# datasets.append(["iris", iris])
# datasets.append(["cmc", cmc])
# datasets.append(["htru_2", htru_2])
# datasets.append(["data_banknote", data_banknote])
# datasets.append(["data_occupancy", data_occupancy])
for dataset in datasets:
    print("\nStarting with " + dataset[0])
    for i in [1, 2, 3, 5, 7, 10]:
        for j in [50, 80, 100, 120, 150, 200, 300, 400, 600]:
            accurancy, test_len, train_len = dataset[1](i, j, 3000)
            tests.append([i, test_len, train_len, accurancy])
    # save data...
    now = datetime.datetime.now()
    resultPath = "results\\Log " + now.strftime("%Y-%m-%d %H %M %S") + ".data"
    with open(resultPath, "w+") as text_file:
        text_file.write("Dataset: " + dataset[0] + "\n")
        text_file.write("Ephocs, Test, Train, Result Accurancy")
        for test in tests:
            print(str(test))
            text_file.write("\n" + str(test).replace("[", "").replace("]", ""))
