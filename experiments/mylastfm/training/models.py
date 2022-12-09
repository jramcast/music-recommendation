from typing import Dict
import numpy as np
import xgboost as xgb
from pandas import DataFrame
from sklearn import linear_model
from sklearn.linear_model._base import LinearModel


def get_model(key: str):
    models: Dict[str, LinearModel] = {
        "xdg": xbg_regressor(),
        "bayes": bayesian_ridge(),
        "baseline": baseline()
    }

    return models[key]


def xbg_regressor():
    return xgb.XGBRegressor(n_estimators=200)


def bayesian_ridge():
    return linear_model.BayesianRidge()


def baseline():
    class BaselineModel:
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

        def save_model(self):
            pass

    return BaselineModel()
