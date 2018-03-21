from PerceptronVoted import PerceptronVoted
import numpy as np


def createDataset1():
    x = np.array([
        [1, 90],
        [2, 45],
        [3, 88],
        [2, 16],
        [1, 8],
        [1, 9],
        [1, 12],
        [1, 19],
        [2, 24],
        [2, 19],
        [3, 50],
        [2, 15],
        [18, 2],
        [40, 3],
        [19, 2],
        [30, 3],
        [26, 2],
        [29, 3],
        [38, 4],
        [59, 1],
        [12, 1],
        [95, 4],
        [32, 2],
        [27, 1],
    ], dtype=float)
    y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ], dtype=float)
    return x, y


x, y = createDataset1()
ephocs = 5
myVoted = PerceptronVoted()
myVoted.train(x, y, ephocs)
to_predict = [[66, 1], [345, 2], [222, 3], [1, 57], [3, 300], [8, 560]]
print("Numbers upper y=x: 1.0, lower y=x: -1.0")
for data_to_predict in to_predict:
    result = myVoted.predict(data_to_predict)
    print("Data: ", data_to_predict, " Received: ", result)
