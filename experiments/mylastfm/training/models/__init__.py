from typing import Dict

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
