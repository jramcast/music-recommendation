from pathlib import Path
from typing import Tuple

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


import dataloading.lastfm
import dataloading.spotify


def print_evaluation_metrics(y_test, y_pred):
    print(" Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print(
        " Root Mean squared error: %.2f"
        % mean_squared_error(y_test, y_pred, squared=False)
    )
    print(" Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print(" Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


if __name__ == "__main__":
    # TODO: Use more features
    # acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,valence
    TARGET_FEATURE = "danceability"

    MODELS_DIR = Path(__file__).parent.joinpath("_models")
    DATA_DIR = Path(__file__).parent.joinpath("../../data/jaime_lastfm")

    lastfm_tag_probs = dataloading.lastfm.load_tag_probs_by_hour(DATA_DIR)
    spotify_features = dataloading.spotify.load_mean_feature_values_by_hour(DATA_DIR)

    dataset = pd.merge(
        lastfm_tag_probs, spotify_features, left_index=True, right_index=True
    )

    # Only predict one FEATURE for now
    X = dataset.iloc[:, :1000]
    y = dataset[TARGET_FEATURE]

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1983
    )  # type: ignore (train_test_split actually returns a dataframe)

    print("Training Set - X size", X_train.shape)
    print("Training Set - y size", y_train.shape)

    model = xgb.XGBRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    print_evaluation_metrics(y_test, y_pred)

    model.save_model(
        MODELS_DIR.joinpath("lastfm-tags_to_spotify-features-xgboost.json")
    )
