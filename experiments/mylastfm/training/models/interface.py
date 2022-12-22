import os
from abc import ABC, abstractmethod, abstractstaticmethod


class Model(ABC):
    """
    Base model interface
    """

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: os.PathLike):
        raise NotImplementedError

    @abstractstaticmethod
    def load(path: os.PathLike):
        raise NotImplementedError
