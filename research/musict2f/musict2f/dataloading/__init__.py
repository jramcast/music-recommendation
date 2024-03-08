from pathlib import Path
from typing import Optional, cast

import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

from musict2f.preprocessing import clean_column_names


def read_tag_probs_sets(
    datadir: Path,
    num_tags: int,
    target: str,
    dimension: str,
    parse_timestamp=False,
    index_col="timestamp",
):
    """
    Returns training validation and test sets, as:

    X_train, y_train, X_validation, y_validation, X_test, Y_test
    """
    dataframe = read_tag_probs(datadir, num_tags, dimension, parse_timestamp, index_col)

    return _split_in_training_validation_and_test(dataframe, num_tags, target)


def read_tag_probs(
    datadir: Path,
    num_tags: int,
    dimension: str,
    parse_timestamp=False,
    index_col="timestamp",
):
    """
    Load the precomputed Last.fm tag probabilites (or weights)
    combined with Spotify features
    """
    df = pd.read_csv(
        datadir.joinpath(f"merged_{num_tags}_tag_probs_by_{dimension}.csv"),
        index_col=index_col,
        parse_dates=[index_col] if parse_timestamp else False,
    )

    # Normalize tag names to avoid problems with xgboost
    # (it has problems with [ or ] or <)
    clean_column_names(df)

    return df


def read_tag_token_sets(
    datadir: Path,
    num_tokens: int,
    target: str,
    dimension: str,
    stringifier_method: str,
    parse_timestamp=False,
    index_col="timestamp",
):
    """
    Returns training validation and test sets, as:

    X_train, y_train, X_validation, y_validation, X_test, Y_test
    """
    dataframe = read_tag_tokens(
        datadir, num_tokens, dimension, stringifier_method, parse_timestamp, index_col
    )

    return _split_in_training_validation_and_test(dataframe, num_tokens, target)


def read_tag_tokens(
    datadir: Path,
    num_tokens: int,
    dimension: str,
    stringifier_method: str,
    parse_timestamp=False,
    index_col="timestamp",
):
    """
    Load the precomputed Last.fm tag tokens
    combined with Spotify features
    """
    return pd.read_csv(
        datadir.joinpath(
            f"merged_{num_tokens}_tokens_from_{stringifier_method}_str"
            f"_by_{dimension}.csv"
        ),
        index_col=index_col,
        parse_dates=[index_col] if parse_timestamp else False,
    )


def read_text_sets(
    datadir: Path,
    dimension: str,
    target_column_name: str,
    stringifier_method: str,
    tokenizer_model_name: str,
    tokenizer_workers: Optional[int] = None,
    block_length=256,
    text_column_name="tags",
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
):
    """
    Load the precomputed Last.fm tags as text
    to be used by the transformers library

    Returns a tuple of datasets.DatasetDict as:
    {
        "train": Dataset,
        "validation": Dataset,
        "test": Dataset,
    }
    """
    train_file = datadir.joinpath(
        f"merged_tag_texts_from_{stringifier_method}_str_by_{dimension}_train.csv"
    )
    validation_file = datadir.joinpath(
        f"merged_tag_texts_from_{stringifier_method}_str_by_{dimension}_validation.csv"
    )
    test_file = datadir.joinpath(
        f"merged_tag_texts_from_{stringifier_method}_str_by_{dimension}_test.csv"
    )

    datasets = cast(
        DatasetDict,
        load_dataset(
            "csv",
            data_files={
                "train": str(train_file),
                "validation": str(validation_file),
                "test": str(test_file),
            },
        ),
    )

    if max_train_samples:
        datasets["train"] = datasets["train"].select(range(max_train_samples))

    if max_validation_samples:
        datasets["validation"] = datasets["validation"].select(
            range(max_validation_samples)
        )

    if max_test_samples:
        datasets["test"] = datasets["test"].select(range(max_test_samples))

    return datasets


def _split_in_training_validation_and_test(
    dataframe: pd.DataFrame,
    num_X_columns: int,
    target_column: str,
):
    X = dataframe.iloc[:, :num_X_columns]
    y = dataframe[target_column]

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_validation: pd.DataFrame
    y_test: pd.DataFrame

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1983
    )  # type: ignore (train_test_split actually returns a dataframe)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.20, random_state=1983
    )  # type: ignore (train_test_split actually returns a dataframe)

    return X_train, y_train, X_validation, y_validation, X_test, y_test
