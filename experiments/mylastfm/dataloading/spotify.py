from pathlib import Path
import pandas as pd


def load_mean_feature_values_by_hour(datadir: Path):
    return pd.read_csv(
        datadir.joinpath("mean_spotify_features_by_hour.csv"), index_col="timestamp"
    )
