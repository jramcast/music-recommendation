from pathlib import Path

import pandas as pd
from preprocessing import clean_column_names


def read_csv_tag_probs(
    datadir: Path, num_tags: int, time_precision: str, parse_timestamp=False
):
    """
    Load the precomputed Last.fm tag probabilites (or weights) by hour
    """
    tagprobs = pd.read_csv(
        datadir.joinpath(f"lastfm_{num_tags}_tag_probs_by_{time_precision}.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"] if parse_timestamp else False,
    )

    # Normalize tag names to avoid problems with xgboost
    # (it has problems with [ or ] or <)
    clean_column_names(tagprobs)

    return tagprobs


def read_csv_tag_tokens(
    datadir: Path,
    num_tokens: int,
    time_precision: str,
    stringifier_method: str,
    parse_timestamp=False,
):
    """
    Load the precomputed Last.fm tag probabilites (or weights) by moment
    """
    tokens = pd.read_csv(
        datadir.joinpath(
            f"lastfm_{num_tokens}_tokens_from_{stringifier_method}_by_{time_precision}.csv"
        ),
        index_col="timestamp",
        parse_dates=["timestamp"] if parse_timestamp else False,
    )

    return tokens


def read_csv_tags_as_text(
    datadir: Path,
    time_precision: str,
    stringifier_method: str,
    parse_timestamp=False,
):
    """
    Load the Last.fm tags as texts by moment
    """
    tokens = pd.read_csv(
        datadir.joinpath(
            f"lastfm_text_from_{stringifier_method}_by_{time_precision}.csv"
        ),
        index_col="timestamp",
        parse_dates=["timestamp"] if parse_timestamp else False,
    )

    return tokens
