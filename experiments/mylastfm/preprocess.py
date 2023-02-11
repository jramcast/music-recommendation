"""
Data preprocessing before training

Run this script before running "train.py" and "recommend.py"
"""
import os
from pathlib import Path
from typing import List

import pandas as pd
from halo import Halo

import preprocessing.lastfm
import preprocessing.spotify
import dataloading.lastfm
import dataloading.spotify


def main():
    TIME_PRECISION = os.environ.get("TIME_PRECISION", "hours")
    TAG_LIMITS = [100, 1000, 10000, 20000]
    TOKEN_LIMITS = [100, 1000, 10000, 20000]
    STRINGIFY_METHODS = ["tag_weight_str", "repeat_tags_str"]

    if os.environ.get("LASTFM_AS_TOKENS", False):
        for limit in TOKEN_LIMITS:
            for method in STRINGIFY_METHODS:
                generate_tokenized_tags_csv(TIME_PRECISION, limit, method)

    if os.environ.get("LASTFM_AS_PROBS", False):
        for limit in TAG_LIMITS:
            generate_tabular_tag_probs_csv(TIME_PRECISION, limit)

    if os.environ.get("SPOTIFY_FEATURES", False):
        generate_spotify_features_csv(TIME_PRECISION)

    if os.environ.get("TEXTS", False):
        for method in STRINGIFY_METHODS:
            generate_texts_csv(TIME_PRECISION, method)

    if os.environ.get("MERGE", False):
        merge_lastfm_and_spotify_csvs(
            TIME_PRECISION, TAG_LIMITS, TOKEN_LIMITS, STRINGIFY_METHODS
        )

    # if os.environ.get("BY_SONG_LASTFM_AS_TOKENS", False):
    #     for limit in TOKEN_LIMITS:
    #         for method in STRINGIFY_METHODS:
    #             generate_tokenized_tags_spotify_features_csv_by_song(limit, method)

    if os.environ.get("BY_SONG_LASTFM_AS_PROBS", False):
        for limit in TAG_LIMITS:
            generate_tabular_tag_probs_and_spotify_features_csv_by_song(limit)

    if os.environ.get("BY_SONG", False):
        for method in STRINGIFY_METHODS:
            generate_tag_texts_and_spotify_features_csv_by_song("asdf")


def generate_tabular_tag_probs_csv(time_precision: str, tags_limit: int):
    """
    Generate a csv of probability values per moment and per tag.
    Tags are not tokenized, preprocessed or stemmized.
    Each different tag is a column
    """
    TAGS_CSV_FILEPATH = Path(__file__).parent.joinpath(
        f"../../data/jaime_lastfm/lastfm_{tags_limit}_tag_probs_by_{time_precision}.csv"
    )

    print("=== Generating dataframe of Last.fm tag relevances by hour...")

    # TODO: clean tags

    df = preprocessing.lastfm.create_tag_probs_by_moment_csv(
        TAGS_CSV_FILEPATH,
        time_precision,
        tags_limit,
    )

    print("")
    print(
        f"Last.fm tags props CSV generated for time precision {time_precision} "
        f"and tags limit{tags_limit}"
    )
    print("Data frame shape", df.shape)
    print(df.head())
    print("Dataframe saved to: ", TAGS_CSV_FILEPATH)


def generate_tokenized_tags_csv(
    time_precision: str, token_limit: int, stringifier_key: str
):
    """
    Generate a csv of tokens, treating tags as natural text and using a tokenizer
    """

    stringifiers = {
        "tag_weight_str": preprocessing.lastfm.tag_stringifier_include_weight,
        "repeat_tags_str": preprocessing.lastfm.tag_stringifier_repeat_tag,
    }

    stringifier = stringifiers[stringifier_key]

    with Halo(text="Generating dataframe of Last.fm tag tokens by moment"):
        TOKENS_CSV_FILEPATH = Path(__file__).parent.joinpath(
            f"../../data/jaime_lastfm/lastfm_{token_limit}_tokens_from_"
            f"{stringifier_key}_by_{time_precision}.csv"
        )

        tokenizer = preprocessing.lastfm.init_tokenizer(token_limit)
        tokens_by_moment = preprocessing.lastfm.create_tag_tokens_by_moment_csv(
            TOKENS_CSV_FILEPATH, tokenizer, stringifier, time_precision
        )

    print("")
    print(
        f"Last.fm tags token CSV generated for time precision {time_precision} "
        f"and token limit {token_limit}"
    )
    print("Data frame shape", tokens_by_moment.shape)
    print("Example")
    print(tokens_by_moment.head())
    print("Dataframe saved to: ", TOKENS_CSV_FILEPATH)


def generate_texts_csv(time_precision: str, stringifier_key: str):
    """
    Generate a csv of texts, concatenating tags
    """

    stringifiers = {
        "tag_weight_str": preprocessing.lastfm.tag_stringifier_include_weight,
        "repeat_tags_str": preprocessing.lastfm.tag_stringifier_repeat_tag,
    }

    stringifier = stringifiers[stringifier_key]

    with Halo(text="Generating dataframe of Last.fm concatenated tags by moment"):
        TOKENS_CSV_FILEPATH = Path(__file__).parent.joinpath(
            f"../../data/jaime_lastfm/lastfm_text_from_"
            f"{stringifier_key}_by_{time_precision}.csv"
        )

        texts_by_moment = preprocessing.lastfm.create_texts_by_moment_csv(
            TOKENS_CSV_FILEPATH, stringifier, time_precision
        )

    print("")
    print(f"Last.fm TEXTS CSV generated for time precision {time_precision} ")
    print("Data frame shape", texts_by_moment.shape)
    print("Example")
    print(texts_by_moment.head())
    print("Dataframe saved to: ", TOKENS_CSV_FILEPATH)


def generate_spotify_features_csv(time_precision: str):
    FILEPATH = Path(__file__).parent.joinpath(
        f"../../data/jaime_lastfm/spotify_features_by_{time_precision}.csv"
    )

    with Halo(text=f"Generating dataframe of Spotify features by {time_precision}"):
        df = preprocessing.spotify.generate_aggregate_features_by_moment_csv(
            FILEPATH, time_precision
        )

        print("")
        print("Spotify features CSV generated for time precision", time_precision)
        print("Data frame shape", df.shape)
        print("Example")
        print(df.head())
        print("Spotify dataframe saved to: ", FILEPATH)


def merge_lastfm_and_spotify_csvs(
    time_precision: str,
    tag_limit_cases: List[int],
    token_limit_cases: List[int],
    stringify_method: List[str],
):
    data_dir = Path(__file__).parent.joinpath("../../data/jaime_lastfm/")
    spotify_features = dataloading.spotify.load_mean_feature_values_by_hour(
        data_dir, time_precision
    )

    for method in stringify_method:

        text_case_name = f"merged_text_from_{method}_by_{time_precision}"

        with Halo(text_case_name):
            lastfm_texts = dataloading.lastfm.read_csv_tags_as_text(
                data_dir, time_precision, method
            )

            dataset = pd.merge(
                lastfm_texts,
                spotify_features,
                left_index=True,
                right_index=True,
            )

            path = data_dir.joinpath(f"{text_case_name}.csv")
            dataset.to_csv(path, index=True)

        for num_tokens in token_limit_cases:
            case_name = f"merged_{num_tokens}_tokens_from_{method}_by_{time_precision}"

            with Halo(case_name):
                lastfm_tag_tokens = dataloading.lastfm.read_csv_tag_tokens(
                    data_dir, num_tokens, time_precision, method
                )

                dataset = pd.merge(
                    lastfm_tag_tokens,
                    spotify_features,
                    left_index=True,
                    right_index=True,
                )

                path = data_dir.joinpath(f"{case_name}.csv")
                dataset.to_csv(path, index=True)

    for num_tags in tag_limit_cases:
        case_name = f"merged_{num_tags}_tag_probs_by_{time_precision}"
        with Halo(case_name):
            lastfm_tag_tokens = dataloading.lastfm.read_csv_tag_probs(
                data_dir, num_tags, time_precision
            )

            dataset = pd.merge(
                lastfm_tag_tokens,
                spotify_features,
                left_index=True,
                right_index=True,
                suffixes=("_x", None),
            )

            path = data_dir.joinpath(f"{case_name}.csv")
            dataset.to_csv(path, index=True)


def generate_tabular_tag_probs_and_spotify_features_csv_by_song(tags_limit: int):
    """
    Generate a csv of probability values per moment and per tag.
    Tags are not tokenized, preprocessed or stemmized.
    Each different tag is a column
    """
    TAGS_CSV_FILEPATH = Path(__file__).parent.joinpath(
        f"../../data/jaime_lastfm/merged_{tags_limit}_tag_probs_by_track.csv"
    )

    print(
        "=== Generating dataframe of Last.fm tag relevances + Spotify features by song..."
    )

    lastfm_tags_by_track = preprocessing.lastfm.get_tag_probs_by_track(
        tags_limit,
    )

    data = []
    # Get all tracks from DB that include audio features
    for track_features in preprocessing.spotify.get_features_for_all_songs():

        key = preprocessing.spotify.get_trackanalysis_key(
            track_features.track_artist, track_features.track_name
        )

        if key in lastfm_tags_by_track:
            # TODO: fix "energy" column conflict
            data.append(
                {
                    "track": key,
                    **lastfm_tags_by_track[key],
                    **track_features.get_features_dict(preprocessing.spotify.FEATURES),
                }
            )

    df = pd.DataFrame(data).set_index("track")
    df.to_csv(TAGS_CSV_FILEPATH, index=True)

    print("")
    print(
        f"Last.fm tags props + Spotify features CSV generated by song "
        f"and tags limit{tags_limit}"
    )
    print("Data frame shape", df.shape)
    print(df.head())
    print("Dataframe saved to: ", TAGS_CSV_FILEPATH)


# TODO Tokenized tags by song
# def generate_tokenized_tags_spotify_features_csv_by_song(
#     time_precision: str, token_limit: int, stringifier_key: str
# ):
#     """
#     Generate a csv of tokens, treating tags as natural text and using a tokenizer
#     """

#     stringifiers = {
#         "tag_weight_str": preprocessing.lastfm.tag_stringifier_include_weight,
#         "repeat_tags_str": preprocessing.lastfm.tag_stringifier_repeat_tag,
#     }

#     stringifier = stringifiers[stringifier_key]

#     with Halo(text="Generating dataframe of Last.fm tag tokens by moment"):
#         TOKENS_CSV_FILEPATH = Path(__file__).parent.joinpath(
#             f"../../data/jaime_lastfm/lastfm_{token_limit}_tokens_from_"
#             f"{stringifier_key}_by_{time_precision}.csv"
#         )

#         tokenizer = preprocessing.lastfm.init_tokenizer(token_limit)
#         tokens_by_moment = preprocessing.lastfm.create_tag_tokens_by_moment_csv(
#             TOKENS_CSV_FILEPATH, tokenizer, stringifier, time_precision
#         )

#     print("")
#     print(
#         f"Last.fm tags token CSV generated for time precision {time_precision} "
#         f"and token limit {token_limit}"
#     )
#     print("Data frame shape", tokens_by_moment.shape)
#     print("Example")
#     print(tokens_by_moment.head())
#     print("Dataframe saved to: ", TOKENS_CSV_FILEPATH)


def generate_tag_texts_and_spotify_features_csv_by_song(stringifier_key: str):
    """
    Generate a csv of last.fm tags as texts and Spotify features, by song
    """

    stringifiers = {
        "tag_weight_str": preprocessing.lastfm.tag_stringifier_include_weight,
        "repeat_tags_str": preprocessing.lastfm.tag_stringifier_repeat_tag,
        "weight_as_order": preprocessing.lastfm.tag_stringifier_weight_as_order,
    }

    lastfm_tags_by_track = preprocessing.lastfm.get_all_last_fm_tags_as_text_by_song(
        stringifiers[stringifier_key]
    )

    data = []

    # Get all tracks from DB that include audio features
    for track_features in preprocessing.spotify.get_features_for_all_songs():

        key = preprocessing.spotify.get_trackanalysis_key(
            track_features.track_artist, track_features.track_name
        )

        if key in lastfm_tags_by_track:
            data.append(
                {
                    "track": key,
                    "tags": lastfm_tags_by_track[key],
                    **track_features.get_features_dict(),
                }
            )

        print(data)
        exit()


if __name__ == "__main__":
    main()
