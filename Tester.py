from PerceptronVoted import PerceptronVoted
from DatasetsFactory import DatasetsFactory
import numpy as np

dataPath = "C:\\Users\\Salimar\\Documents\\AI datasets"


def testData(x, y, ephocs, to_predict, labels):
    print("Ephocs: ", ephocs)
    print("Train: ", len(x))
    print("Test: ", len(to_predict))
    myVoted = PerceptronVoted()
    myVoted.train(x, y, ephocs)
    err = 0
    i = len(to_predict)
    for d, p in zip(to_predict, labels):
        result = myVoted.predict(d)
        if result - p > 0:
            err += 1
        # print("Data: ", d, " predicted: ", result, "Real label:", p)
        # print(i)
        i -= 1
    print("Accurancy: ", (100 - err / len(to_predict)), "%")


def simple(ephocs):
    print("\nsimple dataset")
    x, y = DatasetsFactory.simplePoints()
    to_predict = [[66, 1], [345, 2], [222, 3], [1, 57], [3, 300], [8, 560]]
    labels = [1, 1, 1, -1, -1, -1]
    testData(x, y, ephocs, to_predict, labels)


def wilt(ephocs):
    print("\nWilt Dataset")
    path = dataPath + "\\Wilt"
    x, y = DatasetsFactory.diseasedTrees(path + "\\train.csv")
    to_predict, labels = DatasetsFactory.diseasedTrees(path + "\\test.csv")
    testData(x, y, ephocs, to_predict, labels)


def iris(ephocs):
    print("\niris Dataset")
    basePath = dataPath + "\\iris"
    to_predict, labels = DatasetsFactory.getIris(basePath + "\\test.data")
    x, y = DatasetsFactory.getIris(basePath + "\\train.data")
    testData(x, y, ephocs, to_predict, labels)


def cmc(ephocs):
    print("\nContraceptive Method Choice Dataset")
    basePath = dataPath + "\\cmc"
    to_predict, labels = DatasetsFactory.get_cmc(basePath + "\\test.data")
    x, y = DatasetsFactory.get_cmc(basePath + "\\train.data")
    testData(x, y, ephocs, to_predict, labels)


def movement_aal(ephocs):
    print("\nIndoor User Movement Prediction Dataset")
    basePath = dataPath + "\\MovementAAL"
    x, y, to_predict, labels = DatasetsFactory.getMovement_AAL(basePath, 400)
    testData(x, y, ephocs, to_predict, labels)


def shuttle(ephocs):
    print("\nStatlog (Shuttle) Dataset")
    basePath = dataPath + "\\shuttle"
    to_predict, labels = DatasetsFactory.get_shuttle(basePath + "\\test.data")
    x, y = DatasetsFactory.get_shuttle(basePath + "\\train.data")
    testData(x, y, ephocs, to_predict, labels)


for i in [1, 3, 5, 10]:
    shuttle(i)
