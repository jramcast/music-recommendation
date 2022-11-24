import re
from logging import Logger
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,valence
target_feature="danceability"


def main(feature: str, _run, _log: Logger):

    features = pd.read_csv(
        "data/mean_spotify_features_by_hour.csv", index_col="timestamp"
    )
    tagprobs = pd.read_csv("data/tag_probs_by_hour.csv", index_col="timestamp")

    dataset = pd.merge(tagprobs, features, left_index=True, right_index=True)

    # Normalize tag names to avoid problems with xgboost
    # (it has problems with [ or ] or <)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    dataset.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in dataset.columns.values
    ]

    # Only predict one FEATURE for now
    X = dataset.iloc[:, :1000]
    y = dataset[feature]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1983
    )

    model = xgb.XGBRegressor(n_estimators=200)

    model.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    # The coefficients
    # print("Coefficients: \n", model.coef_)
    # The mean squared error
    _log.info(" Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    _run.log_scalar("Mean squared error", mean_squared_error(y_test, y_pred))

    _log.info(
        " Root Mean squared error: %.2f"
        % mean_squared_error(y_test, y_pred, squared=False)
    )
    _run.log_scalar(
        "Root Mean squared error", mean_squared_error(y_test, y_pred, squared=False)
    )

    _log.info(" Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
    _run.log_scalar("Mean absolute error", mean_absolute_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction
    _log.info(" Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    _run.log_scalar(" Coefficient of determination", r2_score(y_test, y_pred))

    model.save_model("models/lastfm-tags_to_spotify-features-xgboost.json")
