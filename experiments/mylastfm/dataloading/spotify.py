from pathlib import Path
import pandas as pd


def load_features_by_hour(datadir: Path):
    return pd.read_csv(datadir.joinpath("tag_probs_by_hour.csv"), index_col="timestamp")
