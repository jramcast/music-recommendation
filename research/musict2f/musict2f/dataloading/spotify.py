from pathlib import Path
import pandas as pd


def load_mean_feature_values_by_hour(datadir: Path, time_precision: str):
    return pd.read_csv(
        datadir.joinpath(f"spotify_features_by_{time_precision}.csv"),
        index_col="timestamp",
    )
