from PerceptronVoted import PerceptronVoted
from DatasetsFactory import DatasetsFactory
import numpy as np

def test(x, y, ephocs, to_predict):
    myVoted = PerceptronVoted()
    myVoted.train(x, y, ephocs)
    for data_to_predict in to_predict:
        result = myVoted.predict(data_to_predict)
        print("Data: ", data_to_predict, " Received: ", result)


print("\nsimple dataset")
x, y = DatasetsFactory.simplePoints()
to_predict = [[66, 1], [345, 2], [222, 3], [1, 57], [3, 300], [8, 560]]
test(x, y, 5, to_predict)

print("\nrealistic Dataset")
path = "myPath"
to_predict = [[-1]]
x, y = DatasetsFactory.realisticDataset(path)
test(x, y, 5, to_predict)
