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
path1 = "C:\\Users\\Salimar\\Documents\\AI datasets\\devanagariHandwrittenCharacter\\Train\\character_1_ka"
path2 = "C:\\Users\\Salimar\\Documents\\AI datasets\\devanagariHandwrittenCharacter\\Train\\character_3_ga"
to_predict = DatasetsFactory.getImageAsArray("C:\\Users\\Salimar\\Documents\\AI datasets\\devanagariHandwrittenCharacter\\Test\\character_1_ka\\1416.png")
x, y = DatasetsFactory.realisticDataset(path1, path2)
test(x, y, 2, to_predict)
