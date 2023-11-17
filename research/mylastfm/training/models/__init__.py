from typing import Dict

from .interface import Model
from .baseline import BaselineModel
from .xgb import XGBoostRegressor
from .bayes import BayesianRegressor
from .transformer import TransformerRegressor


def get_model(key: str, *args, **kargs):
    models: Dict[str, Model] = {
        "xdg": XGBoostRegressor(),
        "bayes": BayesianRegressor(),
        "baseline": BaselineModel(),
        "gpt2": TransformerRegressor("gpt2", *args, **kargs),
    }

    return models[key]
