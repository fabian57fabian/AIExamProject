from abc import ABC, abstractmethod


class PerceptronAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, x, y, ephocs=1):
        pass

    @abstractmethod
    def predict(self, x):
        pass
