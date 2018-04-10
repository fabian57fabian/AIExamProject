from abc import ABC, abstractmethod


class PerceptronAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def train(self, data, ephocs=1):
        pass

    @abstractmethod
    def predict(self, x):
        pass
