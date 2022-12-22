import numpy as np
from pandas import DataFrame
from .interface import Model


class BaselineModel(Model):

    def fit(self, X: DataFrame, y: DataFrame):
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.y_shape = y.shape

    def predict(self, X: DataFrame):
        num_cases = X.shape[0]
        num_predictions = self.y_shape[1] if len(self.y_shape) > 1 else 1
        return np.random.normal(
            self.y_mean, self.y_std, size=(num_cases, num_predictions)
        )

    def save(self):
        pass

    def load(self):
        pass
