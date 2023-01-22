from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_tag_probs_sets(
    datadir: Path,
    num_tags: int,
    target: str,
    time_precision: str,
    parse_timestamp=False,
):
    """
    Returns training validation and test sets, as:

    X_train, y_train, X_validation, y_validation, X_test, Y_test
    """
    dataframe = read_tag_probs(datadir, num_tags, time_precision, parse_timestamp)

    return _split_in_training_validation_and_test(dataframe, num_tags, target)


def read_tag_probs(
    datadir: Path, num_tags: int, time_precision: str, parse_timestamp=False
):
    """
    Load the precomputed Last.fm tag probabilites (or weights)
    combined with Spotify features
    """
    return pd.read_csv(
        datadir.joinpath(f"merged_{num_tags}_tag_probs_by_{time_precision}.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"] if parse_timestamp else False,
    )


def read_tag_token_sets(
    datadir: Path,
    num_tokens: int,
    target: str,
    time_precision: str,
    stringifier_method: str,
    parse_timestamp=False,
):
    """
    Returns training validation and test sets, as:

    X_train, y_train, X_validation, y_validation, X_test, Y_test
    """
    dataframe = read_tag_tokens(
        datadir, num_tokens, time_precision, stringifier_method, parse_timestamp
    )

    return _split_in_training_validation_and_test(dataframe, num_tokens, target)


def read_tag_tokens(
    datadir: Path,
    num_tokens: int,
    time_precision: str,
    stringifier_method: str,
    parse_timestamp=False,
):
    """
    Load the precomputed Last.fm tag tokens
    combined with Spotify features
    """
    return pd.read_csv(
        datadir.joinpath(
            f"merged_{num_tokens}_tokens_from_{stringifier_method}_str"
            f"_by_{time_precision}.csv"
        ),
        index_col="timestamp",
        parse_dates=["timestamp"] if parse_timestamp else False,
    )


def read_text(
    datadir: Path,
    time_precision: str,
    stringifier_method: str,
    parse_timestamp=False,
):
    """
    Load the precomputed Last.fm tags joined as simple text
    """
    return pd.read_csv(
        datadir.joinpath(
            f"merged_text_from_{stringifier_method}_str"
            f"_by_{time_precision}.csv"
        ),
        index_col="timestamp",
        parse_dates=["timestamp"] if parse_timestamp else False,
    )


def read_text_sets(
    datadir: Path,
    time_precision: str,
    stringifier_method: str,
    target: str,
    parse_timestamp=False,
):

    """
    Returns training validation and test sets for the texts dataset, as:

    X_train, y_train, X_validation, y_validation, X_test, Y_test
    """
    dataframe = read_text(
        datadir, time_precision, stringifier_method, parse_timestamp
    )

    return _split_in_training_validation_and_test(dataframe, 1, target)


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
