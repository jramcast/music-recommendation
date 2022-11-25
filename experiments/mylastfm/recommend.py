"""
Run the recommendations pipeline
"""

from pathlib import Path
from pprint import pprint
from datetime import datetime

import xgboost as xgb
from pymongo import MongoClient

import dataloading.lastfm


def get_lastfm_tag_probs_for_current_hour(lastfm_tag_probs):
    def get_hour_from_datetime(moment: datetime):
        return moment.hour

    # Get input last fm tags for the current hour of the day
    lastfm_tag_probs["hour"] = lastfm_tag_probs.index.map(get_hour_from_datetime)
    tagprobs_aggregate_by_hour = lastfm_tag_probs.groupby(["hour"]).mean()

    # Get input last fm tags for the current hour of the day
    current_hour = datetime.now().hour
    current_hour_tag_probs = tagprobs_aggregate_by_hour.iloc[current_hour].to_frame().T

    return current_hour, current_hour_tag_probs


def predict_spotify_features(model_path: Path, input_tags):
    # Load the pretrained model to predict Spotify features
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    prediction = model.predict(input_tags)
    return prediction


def find_closest_tracks_to_spotify_features(danceability):
    client = MongoClient()
    db = client.mgr

    results_below = (
        db.spotify_audiofeatures.find(
            {"features.danceability": {"$lte": danceability}},
            {"features": True, "track_name": True, "track_artist": True},
        )
        .sort("features.danceability", -1)
        .limit(10)
    )

    results_above = (
        db.spotify_audiofeatures.find(
            {"features.danceability": {"$gt": danceability}},
            {"features": True, "track_name": True, "track_artist": True},
        )
        .sort("features.danceability", 1)
        .limit(10)
    )

    results = []

    for doc in results_below:
        results.append(
            {
                "track": doc["track_name"],
                "artist": doc["track_artist"],
                "danceability": doc["features"][0]["danceability"],
                "distance": abs(doc["features"][0]["danceability"] - danceability),
            }
        )

    for doc in results_above:
        results.append(
            {
                "track_name": doc["track_name"],
                "track_artist": doc["track_artist"],
                "danceability": doc["features"][0]["danceability"],
                "distance": abs(doc["features"][0]["danceability"] - danceability),
            }
        )

    def sort_by_distance(doc):
        return doc["distance"]

    results.sort(key=sort_by_distance)
    return results


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.joinpath("../..//data/jaime_lastfm")
    MODEL_PATH = Path(__file__).parent.joinpath(
        "_models/lastfm-tags_to_spotify-features-xgboost.json"
    )

    lastfm_tag_probs = dataloading.lastfm.load_tag_probs_by_hour(DATA_DIR)
    hour, lastfm_tag_probs = get_lastfm_tag_probs_for_current_hour(lastfm_tag_probs)
    print("🎶 PHASE 1: Generating Last.fm input tag scores for current time 🎶")
    print("  - Current hour is:", hour)
    print(
        "  - The following values are an average of the "
        "strength of each tag for the current hour:\n"
    )
    print(lastfm_tag_probs)

    print(
        "\n\n🤖 PHASE 2: Predicting Spotify high-level"
        " features from Last.fm tag scores 🤖\n"
    )
    prediction = predict_spotify_features(MODEL_PATH, lastfm_tag_probs)
    print("  - Predicted Spotify features")
    danceability = prediction[0].item()
    print("    * Danceability:", danceability)

    print(
        "\n\n🔝 PHASE 3: Ranking. Find tracks closest"
        " to the predicted Spotify features 🔝"
    )

    ranking = find_closest_tracks_to_spotify_features(danceability)
    pprint(ranking)
