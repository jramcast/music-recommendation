import os
import time
from pathlib import Path

import logging
from typing import Dict

import evaluation
import training.models
import dataloading


training_metrics = evaluation.TrainingMetrics(
    Path(__file__).parent.joinpath("normalized_results_by_track.csv")
)


def main():
    DEFAULT_DATA_DIR = Path(__file__).parent.joinpath("../../data/jaime_lastfm")
    DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
    MODELS_SAVE_DIR = Path(__file__).parent.joinpath("_normalized_models")
    TARGET_FEATURE = os.getenv("TARGET_FEATURE")
    if TARGET_FEATURE:
        TARGET_FEATURES = [TARGET_FEATURE]
    else:
        TARGET_FEATURES = [
            "danceability",
            "acousticness",
            "energy",
            "instrumentalness",
            "valence",
        ]
    TABULAR_MODELS = ["bayes", "xgboost"]
    DIMENSIONS = ["track"]

    configure_logging()

    NUM_TAGS = [1000]

    for num_tokens in NUM_TAGS:
        for dimension in DIMENSIONS:
            for target in TARGET_FEATURES:
                for model in TABULAR_MODELS:
                    train_with_tag_probs(
                        target,
                        model,
                        num_tokens,
                        dimension,
                        MODELS_SAVE_DIR,
                        DATA_DIR,
                    )


def train_with_tag_probs(
    target: str,
    model_key: str,
    num_tags: int,
    dimension: str,
    models_save_dir: Path,
    data_dir: Path,
):
    experiment = f"{target}-{model_key}-{num_tags}-probs-by_{dimension}"

    logger = logging.getLogger(experiment)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _,
        _,
    ) = dataloading.read_tag_probs_sets(
        data_dir, num_tags, target, dimension, index_col=dimension
    )

    print(X_train)

    logger.info(f"Training Set size X: {X_train.shape}")
    logger.info(f"Training Set size y: {y_train.shape}")

    if model_key == "bayes":
        model = training.models.BayesianRegressor()
    elif model_key == "xgboost":
        model = training.models.XGBoostRegressor()

    start = time.time()
    logger.info("Start training")
    model.fit(X_train, y_train)
    logger.info(f"Training time: {time.time() - start} seconds")

    y_pred = model.predict(X_validation)
    experiment_metrics = training_metrics.evaluate(experiment, y_validation, y_pred)
    log_metrics(logger, experiment_metrics)

    model.save(models_save_dir.joinpath(f"{experiment}.json"))


def configure_logging():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # handlers=[
        #     logging.FileHandler(Path(__file__).parent.joinpath("results_by_track.log")),
        #     logging.StreamHandler(),
        # ],
    )


def log_metrics(logger: logging.Logger, metrics: Dict):
    logger.info("Mean squared error: %.2f" % metrics["mse"])
    logger.info("Root Mean squared error: %.2f" % metrics["rmse"])
    logger.info("Mean absolute error: %.2f" % metrics["mae"])
    logger.info("Coefficient of determination: %.2f" % metrics["r2"])


if __name__ == "__main__":
    main()
