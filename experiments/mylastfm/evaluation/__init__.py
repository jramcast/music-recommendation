import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class TrainingMetrics:

    """
    Logs training metrics into a dataframe and saves it to a CSV file
    """

    results: Dict

    def __init__(self, csv_path: Path) -> None:
        self.results = {}
        self.csv_path = csv_path

    def evaluate(self, experiment_name: str, y_test, y_pred):
        self.results[experiment_name] = calculate_metrics(y_test, y_pred).values()
        self.to_csv()

        return self.results[experiment_name]

    def to_csv(self):
        df = pd.DataFrame.from_dict(
            self.results, orient="index", columns=["mse", "rmse", "mae", "r2"]
        )
        df.to_csv()


def calculate_metrics(y_test, y_pred):
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
        # The coefficient of determination: 1 is perfect prediction
        "r2": r2_score(y_test, y_pred),
    }

