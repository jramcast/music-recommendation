import os

import json
import pandas as pd
from pathlib import Path
from joblib import load
import numpy as np
import xgboost


MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).parent / "_models"))
DATA_DIR = "normalized_data"


def load_xgboost_model(filepath: str):
    m = xgboost.XGBRegressor(n_estimators=200)
    m.load_model(filepath)
    return m


features = ["acousticness", "danceability", "energy", "instrumentalness", "valence"]
scalers = {f: load(MODELS_DIR / f"scaler_{f}.joblib") for f in features}
models = {
    "bayes": {
        f: load(MODELS_DIR / f"{f}-bayes-1000-probs-by_track.json") for f in features
    },
    "xgboost": {
        f: load_xgboost_model(MODELS_DIR / f"{f}-xgboost-1000-probs-by_track.json")
        for f in features
    },
}


with open(Path(__file__).parent / "top-1000-tags.json") as f:
    columns = json.loads(f.read())

tags = [
    ["electronic", "100"],
    ["electro", "60"],
    ["french", "56"],
    ["dance", "52"],
    ["seen live", "45"],
    ["electronica", "33"],
    ["House", "15"],
    ["ed banger", "9"],
    ["techno", "7"],
    ["electro house", "5"],
    ["new rave", "5"],
    ["hardcore", "4"],
    ["french house", "4"],
    ["Electroclash", "4"],
    ["french electro", "4"],
    ["justice", "3"],
    ["indie", "3"],
    ["france", "3"],
    ["electropop", "2"],
    ["Disco", "2"],
    ["00s", "2"],
    ["french touch", "2"],
    ["dance punk", "2"],
    ["hardcore punk", "2"],
    ["funk", "2"],
    ["alternative", "1"],
    ["thrash metal", "1"],
    ["experimental", "1"],
    ["party", "1"],
    ["indietronica", "1"],
    ["club", "1"],
    ["want to see live", "1"],
]


def build_input(tags):
    sample = [0] * 1000

    for tag, weight in tags:
        try:
            index = columns.index(tag)
        except:
            continue

        sample[index] = int(weight)

    df = pd.DataFrame([sample], columns=columns)

    print(df)

    return df


def inverse_feature_transform(value, feature):
    # Inverse transform to get back the normalized target variable data
    normal_data = scalers[feature].inverse_transform(np.array(value).reshape(-1, 1))

    # Reverse the log transformation
    value = np.exp(normal_data) - 0.0001

    # Inverse of the normalization
    value = value / (1 + value)

    return value[0][0]


input = build_input(tags)

results = {}

for f in features:
    value = models[f].predict(input)
    print(f"resulted {f}", value)
    results[f] = inverse_feature_transform(value, f)


print(results)
