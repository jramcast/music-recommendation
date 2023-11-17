import os
import xgboost
from .interface import Model


class XGBoostRegressor(Model):

    def __init__(self) -> None:
        self._model = xgboost.XGBRegressor(n_estimators=200)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def save(self, path: os.PathLike):
        self._model.save_model(path)

    def load(self, path: os.PathLike):
        self._model.load_model(path)
