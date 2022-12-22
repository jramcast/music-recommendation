import os
import joblib
from sklearn import linear_model
from .interface import Model


class BayesianRegressor(Model):

    def __init__(self) -> None:
        self._model = linear_model.BayesianRidge()

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def save(self, path: os.PathLike):
        joblib.dump(self._model, path)
