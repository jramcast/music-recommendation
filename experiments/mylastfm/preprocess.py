"""
Data preprocessing before training

Run this script before running "train.py" and "recommend.py"
"""
import os
from pathlib import Path

from halo import Halo
import preprocessing.lastfm


if __name__ == "__main__":
    TIME_PRECISION = "hours"

    if os.environ.get("GENERATE_TAG_PROBS_CSV"):
        TAGS_LIMIT = 10000

        TAGS_CSV_FILEPATH = Path(__file__).parent.joinpath(
            f"../../data/jaime_lastfm/{TAGS_LIMIT}_tag_probs_by_{TIME_PRECISION}.csv"
        )

        print("=== Generating dataframe of Last.fm tag relevances by hour...")

        # TODO: clean tags
        # TODO: maybe use alternative representation (bag of words?)

        df = preprocessing.lastfm.create_tag_probs_by_moment_csv(
            TAGS_CSV_FILEPATH,
            TIME_PRECISION,
            TAGS_LIMIT,
        )

        print("")
        print("Generated data frame shape", df.shape)
        print("Example")
        print(df.head())
        print("Dataframe saved to: ", TAGS_CSV_FILEPATH)

    # TODO: generate spotify CSV...

    with Halo(text="Generating dataframe of Last.fm tag tokens by moment"):
        TOKENS_CSV_FILEPATH = Path(__file__).parent.joinpath(
            f"../../data/jaime_lastfm/tag_tokens_by_{TIME_PRECISION}.csv"
        )

        tokenizer = preprocessing.lastfm.create_tokenizer()
        tokens_by_moment = preprocessing.lastfm.create_tag_tokens_by_moment_csv(
            TOKENS_CSV_FILEPATH, tokenizer
        )

    print("Generated data frame shape", tokens_by_moment.shape)
    print("Example")
    print(tokens_by_moment.head())
    print("Dataframe saved to: ", TOKENS_CSV_FILEPATH)
