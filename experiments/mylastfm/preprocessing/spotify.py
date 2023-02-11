from pathlib import Path
import statistics
from typing import Dict, Union

import pandas as pd
from pymongo import MongoClient

from dataloading.raw.lastfm import MongoDBTrackPlaysRepository
from dataloading.raw.spotify import MongoDBSpotifyAudioFeaturesRepository


FEATURES = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "valence",
]


def get_features_for_all_songs():
    client = MongoClient()
    db = client.mgr
    features_repo = MongoDBSpotifyAudioFeaturesRepository(db.spotify_audiofeatures)

    for track in features_repo.all_features():
        if track.has_features():
            yield track



def generate_aggregate_features_by_moment_csv(
    csv_filename: Path, time_precision="hours"
):

    client = MongoClient()
    db = client.mgr

    moments = {}

    trackplays_repo = MongoDBTrackPlaysRepository(db.lastfm_playedtracks)
    features_repo = MongoDBSpotifyAudioFeaturesRepository(db.spotify_audiofeatures)

    # Load spotify features in memory
    features_by_track = {}
    features_cursor = features_repo.all_features()
    for f in features_cursor:
        key = get_trackanalysis_key(f.track_artist, f.track_name)
        features_by_track[key] = f.features[0]

        if len(f.features) > 1:
            print(key, f.features)

    for track in trackplays_repo.all():

        trackanalysis_key = get_trackanalysis_key(track.artist.name, track.name)
        trackfeatures = features_by_track.get(trackanalysis_key)

        if not trackfeatures:
            continue

        # Round time for aggregation
        moment = track.playback_utc_date.isoformat(timespec=time_precision)

        if moment not in moments:
            moments[moment] = []

        # Only use FEATURES
        trackfeatures = {k: trackfeatures[k] for k in FEATURES}

        moments[moment].append(trackfeatures)

    # Aggregate cases for each moment (AVERAGE)
    moments_aggregated = {}
    moments_no_data = 0
    for moment, trackplays_features in moments.items():
        aggregates = {}
        values = {}

        if len(trackplays_features) == 0:
            moments_no_data += 1
            continue

        for feature in FEATURES:
            for features_for_track in trackplays_features:
                values[feature] = values.get(feature, []) + [
                    features_for_track[feature]
                ]

            aggregates[feature] = statistics.mean(values[feature])

        # also prepare for pandas
        moments_aggregated[moment] = {"timestamp": moment, **aggregates}

    df = pd.DataFrame(moments_aggregated.values())
    df.to_csv(csv_filename, index=False)

    return df


def get_trackanalysis_key(artist, name):
    return f"{artist} - {name}"
