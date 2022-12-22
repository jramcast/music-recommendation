from typing import Dict

import xgboost as xgb
from sklearn import linear_model

from .interface import Model
from .baseline import BaselineModel
from .xgb import XGBoostRegressor
from .bayes import BayesianRegressor


def get_model(key: str):
    models: Dict[str, Model] = {
        "xdg": XGBoostRegressor(),
        "bayes": BayesianRegressor(),
        "baseline": BaselineModel()
    }

    return models[key]
