import os
import time
from pathlib import Path

import logging
from typing import Dict

import evaluation
import training.models
import dataloading


training_metrics = evaluation.TrainingMetrics(
    Path(__file__).parent.joinpath("results_by_track.csv")
)


def main():
    DEFAULT_DATA_DIR = Path(__file__).parent.joinpath("../../data/jaime_lastfm")
    DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
    MODELS_SAVE_DIR = Path(__file__).parent.joinpath("_models")
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
    TABULAR_MODELS = ["xdg", "bayes", "baseline"]
    TRANSFORMER_MODELS = ["gpt2"]
    DIMENSIONS = ["track"]

    configure_logging()

    if os.environ.get("TRAIN_PROBS", False):
        NUM_TAGS = [100, 1000, 10000]

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

    if os.environ.get("TRAIN_TOKENS", False):
        NUM_TOKENS = [100, 1000, 10000]
        STRING_METHOD = os.getenv("STRING_METHOD")
        if STRING_METHOD:
            STRING_METHODS = [STRING_METHOD]
        else:
            STRING_METHODS = ["tag_weight", "repeat_tags", "tag_order"]

        for num_tokens in NUM_TOKENS:
            for dimension in DIMENSIONS:
                for stringifier_method in STRING_METHODS:
                    for target in TARGET_FEATURES:
                        for model in TABULAR_MODELS:
                            train_with_tag_tokens(
                                target,
                                model,
                                num_tokens,
                                dimension,
                                stringifier_method,
                                MODELS_SAVE_DIR,
                                DATA_DIR,
                            )

    if os.environ.get("TRAIN_TEXTS", True):
        STRING_METHODS = ["tag_weight", "repeat_tags", "tag_order"]

        for dimension in DIMENSIONS:
            for stringifier_method in STRING_METHODS:
                for target in TARGET_FEATURES:
                    for model in TRANSFORMER_MODELS:
                        train_with_tag_texts(
                            target,
                            model,
                            dimension,
                            stringifier_method,
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
    experiment = f"{target}-{model_key}-{num_tags}_probs-by_{dimension}"

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
    dimension: str,
    stringifier_method: str,
    models_save_dir: Path,
    data_dir: Path,
):
    experiment = (
        f"{target}-{model_key}-{num_tokens}_tokens-"
        f"from_{stringifier_method}-by_{dimension}"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _,
        _,
    ) = dataloading.read_tag_token_sets(
        data_dir, num_tokens, target, dimension, stringifier_method, index_col=dimension
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

    model.save(models_save_dir.joinpath(f"{experiment}.json"))


def train_with_tag_texts(
    target: str,
    model_key: str,
    dimension: str,
    stringifier_method: str,
    models_save_dir: Path,
    data_dir: Path,
):
    experiment = (
        f"{target}-{model_key}-_tag_texts-" f"from_{stringifier_method}-by_{dimension}"
    )

    datasets = dataloading.read_text_sets(
        data_dir,
        dimension,
        target,
        stringifier_method,
        tokenizer_model_name=model_key,
        # Uncomment to use a subset of the data
        # max_train_samples=5,
        # max_validation_samples=5,
        # max_test_samples=5,
    )

    logger = logging.getLogger(experiment)
    logger.info("Training Set: " + str(datasets["train"].shape))
    logger.info("Validation Set: " + str(datasets["validation"].shape))
    logger.info("Test Set: " + str(datasets["test"].shape))

    model_save_dir = models_save_dir.joinpath(experiment)

    model = training.models.get_model(
        model_key,
        target_column_name=target,
        output_dir=model_save_dir,
        logger=logger,
        train_epochs=10,
        train_batch_size=10,
        eval_batch_size=10,
        # Uncomment to run a smaller train phase 
        # train_save_steps=200,
        # eval_steps=100,
    )

    model.fit(datasets)

    # TODO: evaluate


def configure_logging():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(__file__).parent.joinpath("results_by_track.log")),
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
