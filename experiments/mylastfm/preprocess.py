"""
Data preprocessing before training

Run this script before running "train.py" and "recommend.py"
"""
from pathlib import Path
import preprocessing.lastfm


if __name__ == "__main__":
    # Moments are computed according to the precision
    TAGS_LIMIT = 10000
    TIME_PRECISION = "hours"

    CSV_FILEPATH = Path(__file__).parent.joinpath(
            f"../../data/jaime_lastfm/{TAGS_LIMIT}_tag_probs_by_{TIME_PRECISION}.csv"
        )

    print("Generating dataframe of Last.fm tag relevances by hour...")

    # TODO: clean tags
    # TODO: maybe use alternative representation (bag of words?)

    df = preprocessing.lastfm.create_tag_probs_by_moment_csv(
        CSV_FILEPATH,
        TIME_PRECISION,
        TAGS_LIMIT,
    )

    print("")
    print("Generated data frame shape", df.shape)
    print("Example")
    print(df.head())
    print("Dataframe saved to: ", CSV_FILEPATH)

    # TODO: generate spotify CSV...
