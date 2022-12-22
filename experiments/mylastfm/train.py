import os
import time
from pathlib import Path

import logging
from typing import Dict

import evaluation
import training.models
import dataloading


training_metrics = evaluation.TrainingMetrics(
    Path(__file__).parent.joinpath("results.csv")
)


def main():
    DATA_DIR = Path(__file__).parent.joinpath("../../data/jaime_lastfm")
    MODELS_SAVE_DIR = Path(__file__).parent.joinpath("_models")
    # TODO: Use more features
    # acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,valence
    TARGET_FEATURES = [
        "danceability",
        "acousticness",
        "energy",
        "instrumentalness",
        "valence",
    ]
    MODELS = [
        "xdg",
        "bayes",
        "baseline"
    ]
    TIME_PRECISIONS = ["hours"]

    configure_logging()

    if os.environ.get("TRAIN_PROBS", True):
        NUM_TAGS = [100, 1000, 10000]

        for num_tokens in NUM_TAGS:
            for time_precision in TIME_PRECISIONS:
                for target in TARGET_FEATURES:
                    for model in MODELS:
                        train_with_tag_probs(
                            target,
                            model,
                            num_tokens,
                            time_precision,
                            MODELS_SAVE_DIR,
                            DATA_DIR,
                        )

    if os.environ.get("TRAIN_TOKENS", True):
        NUM_TOKENS = [100, 1000, 10000]
        STRING_METHODS = ["tag_weight", "repeat_tags"]

        for num_tokens in NUM_TOKENS:
            for time_precision in TIME_PRECISIONS:
                for stringifier_method in STRING_METHODS:
                    for target in TARGET_FEATURES:
                        for model in MODELS:
                            train_with_tag_tokens(
                                target,
                                model,
                                num_tokens,
                                time_precision,
                                stringifier_method,
                                MODELS_SAVE_DIR,
                                DATA_DIR,
                            )


def train_with_tag_probs(
    target: str,
    model_key: str,
    num_tags: int,
    time_precision: str,
    models_save_dir: Path,
    data_dir: Path,
):
    experiment = f"{target}-{model_key}-{num_tags}_probs-by_{time_precision}"

    logger = logging.getLogger(experiment)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _,
        _,
    ) = dataloading.read_tag_probs_sets(data_dir, num_tags, target, time_precision)

    logger.info(f"Training Set size X: {X_train.shape}")
    logger.info(f"Training Set size y: {y_train.shape}")

    model = training.models.get_model(model_key)

    start = time.time()
    logger.info("Start training")
    model.fit(X_train, y_train)
    logger.info(f"Training time: {time.time() - start} seconds")

    y_pred = model.predict(X_validation)
    experiment_metrics = training_metrics.evaluate(experiment, y_validation, y_pred)
    log_metrics(logger, experiment_metrics)

    model.save(models_save_dir.joinpath(f"{experiment}.json"))


def train_with_tag_tokens(
    target: str,
    model_key: str,
    num_tokens: int,
    time_precision: str,
    stringifier_method: str,
    models_save_dir: Path,
    data_dir: Path,
):
    experiment = (
        f"{target}-{model_key}-{num_tokens}_tokens-"
        f"from_{stringifier_method}-by_{time_precision}"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _,
        _,
    ) = dataloading.read_tag_token_sets(
        data_dir, num_tokens, target, time_precision, stringifier_method
    )

    logger = logging.getLogger(experiment)
    logger.info(f"Training Set size X: {X_train.shape}")
    logger.info(f"Training Set size y: {y_train.shape}")

    model = training.models.get_model(model_key)

    start = time.time()
    logger.info("Start training")
    model.fit(X_train, y_train)
    logger.info(f"Training time: {time.time() - start} seconds")

    y_pred = model.predict(X_validation)
    experiment_metrics = training_metrics.evaluate(experiment, y_validation, y_pred)
    log_metrics(logger, experiment_metrics)
    models_save_dir.joinpath(f"{experiment}.json")


def configure_logging():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(__file__).parent.joinpath("results.log")),
            logging.StreamHandler(),
        ],
    )


def log_metrics(logger: logging.Logger, metrics: Dict):
    logger.info("Mean squared error: %.2f" % metrics["mse"])
    logger.info("Root Mean squared error: %.2f" % metrics["rmse"])
    logger.info("Mean absolute error: %.2f" % metrics["mae"])
    logger.info("Coefficient of determination: %.2f" % metrics["r2"])


if __name__ == "__main__":
    main()
