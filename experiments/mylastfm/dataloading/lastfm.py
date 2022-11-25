from pathlib import Path

import pandas as pd
from preprocessing import clean_column_names


def load_tag_probs_by_hour(datadir: Path):
    """
    Load the precomputed Last.fm tag probabilites (or weights) by hour
    """
    tagprobs = pd.read_csv(
        datadir.joinpath("tag_probs_by_hour.csv"),
        index_col="timestamp",
        parse_dates=["timestamp"],
    )

    # Normalize tag names to avoid problems with xgboost
    # (it has problems with [ or ] or <)
    clean_column_names(tagprobs)

    return tagprobs
