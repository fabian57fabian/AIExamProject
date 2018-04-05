from PerceptronVoted import PerceptronVoted
from DatasetsFactory import DatasetsFactory
import datetime


def abs(number):
    if number == 0:
        return number
    if number < 0:
        return number * -1
    return number


def testData(x, y, ephocs, to_predict, labels):
    print("Train: ", len(x), "Test: ", len(to_predict), "Ephocs: ", ephocs)
    myVoted = PerceptronVoted()
    myVoted.train(x, y, ephocs)
    err = 0
    i = len(to_predict)
    for d, p in zip(to_predict, labels):
        result = myVoted.predict(d)
        if abs(result - p) != 0:
            err += 1
        # print("Element: ", i, "Data: ", d, " predicted: ", result, "Real label:", p)
        i -= 1
    errors = (err / len(to_predict) * 100)
    print("Accurancy: ", 100 - errors, "%", "Errors: ", errors, "%")
    return 100 - errors


def simple(ephocs):
    dataset_name = "simple"
    x, y = DatasetsFactory.simplePoints()
    to_predict = [[66, 1], [345, 2], [222, 3], [1, 57], [3, 300], [8, 560]]
    labels = [1, 1, 1, -1, -1, -1]
    testData(x, y, ephocs, to_predict, labels)


def wilt(ephocs):
    dataset_name = "wilt"
    path = dataPath + "\\Wilt"
    x, y = DatasetsFactory.diseasedTrees(path + "\\train.csv")
    to_predict, labels = DatasetsFactory.diseasedTrees(path + "\\test.csv")
    testData(x, y, ephocs, to_predict, labels)


def iris(ephocs):
    dataset_name = "iris"
    basePath = dataPath + "\\iris"
    to_predict, labels = DatasetsFactory.getIris(basePath + "\\test.data")
    x, y = DatasetsFactory.getIris(basePath + "\\train.data")
    testData(x, y, ephocs, to_predict, labels)


def cmc(ephocs):
    dataset_name = "cmc"
    basePath = dataPath + "\\cmc"
    to_predict, labels = DatasetsFactory.get_cmc(basePath + "\\test.data")
    x, y = DatasetsFactory.get_cmc(basePath + "\\train.data")
    testData(x, y, ephocs, to_predict, labels)


def movement_aal(ephocs):
    dataset_name = "movement_aal"
    basePath = dataPath + "\\MovementAAL"
    x, y, to_predict, labels = DatasetsFactory.getMovement_AAL(basePath, 400)
    testData(x, y, ephocs, to_predict, labels)


def shuttle(ephocs):
    dataset_name = "shuttle"
    basePath = dataPath + "\\shuttle"
    to_predict, labels = DatasetsFactory.get_shuttle(basePath + "\\test.data")
    x, y = DatasetsFactory.get_shuttle(basePath + "\\train.data")
    testData(x, y, ephocs, to_predict, labels)


# not working. Really not lineary separable
def bank_marketing(ephocs):
    dataset_name = "bank_marketing"
    basePath = "datasets" + "\\BankMarketing"
    to_predict, labels = DatasetsFactory.bankMarketing(basePath + "\\bank-additional-full.csv", 500, 500)
    x, y = DatasetsFactory.bankMarketing(basePath + "\\bank-additional.csv", 2000, 2000)
    testData(x, y, ephocs, to_predict, labels)


def eye_state(ephocs, train, test):
    basePath = "datasets" + "\\EEG_Eye_State"
    to_predict, labels = DatasetsFactory.eye_state(basePath + "\\EEG Eye State.arff", test / 2,
                                                   test / 2)
    x, y = DatasetsFactory.eye_state(basePath + "\\EEG Eye State.arff", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(y)


def data_banknote(ephocs, train, test):
    basePath = "datasets" + "\\banknote_authentication"
    to_predict, labels = DatasetsFactory.data_banknote(basePath + "\\data_banknote_authentication.txt", test / 2,
                                                       test / 2)
    x, y = DatasetsFactory.data_banknote(basePath + "\\data_banknote_authentication.txt", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(y)


def data_occupancy(ephocs, train, test):
    basePath = "datasets" + "\\occupancy_data"
    to_predict, labels = DatasetsFactory.data_occupancy(basePath + "\\datatest.txt", test / 2,
                                                        test / 2)
    x, y = DatasetsFactory.data_occupancy(basePath + "\\datatraining.txt", train / 2, train / 2)
    return testData(x, y, ephocs, to_predict, labels), len(y)


tests = []
datasets = []
# datasets.append(["eye_state", eye_state])
datasets.append(["data_banknote", data_banknote])
datasets.append(["data_occupancy", data_occupancy])
for dataset in datasets:
    print("\nStarting with " + dataset[0])
    for i in [1, 2, 3, 5, 7, 10]:
        for j in [50, 80, 100, 120, 150, 200, 300, 400, 600]:
            accurancy, test_len = dataset[1](i, j, 4000)
            tests.append([i, j, test_len, accurancy])
    # save data...
    now = datetime.datetime.now()
    resultPath = "results\\Log " + now.strftime("%Y-%m-%d %H %M %S") + ".data"
    with open(resultPath, "w+") as text_file:
        text_file.write("Dataset: " + dataset[0]+"\n")
        text_file.write("Ephocs, Test, Train, Result Accurancy")
        for test in tests:
            print(str(test))
            text_file.write("\n" + str(test).replace("[", "").replace("]", ""))
