"""
Last.fm data preprocessing before training.

This scripts reads the raw Last.fm from a MongoDB and creates a CSV
"""

from typing import Dict, Tuple, Iterable, List

import pandas as pd
from pymongo import MongoClient
from tokenizers import Tokenizer, Encoding
from transformers import AutoTokenizer


from dataloading.raw.entities.track import Track
from dataloading.raw.lastfm import MongoDBTrackPlaysRepository


def create_tag_probs_by_moment_csv(
    csv_filename, time_precision="hours", tags_limit=999999999
):
    """
    Generates a CSV file of Last.fm tag relevances by hour
    """
    trackplays = get_all_lastfm_track_plays()

    moment_to_tags_mapping = map_moments_to_tags(trackplays, time_precision)

    tag_relevances = create_tags_dataframe_sorted_by_relevance(moment_to_tags_mapping)

    print("Num unique tags", tag_relevances.shape[0])

    # Limit number of tags to include
    tag_relevances = tag_relevances[:tags_limit]

    # Get the list of tags
    tagnames = tag_relevances.index.tolist()

    # Drop duplicates and sum weights
    moments_to_tag_relevances_mapping = calculate_tag_relevance_by_moment(
        moment_to_tags_mapping, tagnames
    )

    # Create the dataframe
    moments_to_tag_relevances_df = create_moments_to_tags_relevance_df(
        tagnames, moments_to_tag_relevances_mapping
    )

    moments_to_tag_relevances_df.to_csv(csv_filename, index=False)

    return moments_to_tag_relevances_df


def get_all_lastfm_track_plays():
    client = MongoClient()
    db = client.mgr
    repo = MongoDBTrackPlaysRepository(db.lastfm_playedtracks)
    trackplays = repo.all()
    return trackplays


MomentTags = Dict[str, List[Tuple[str, str]]]


def map_moments_to_tags(trackplays: Iterable[Track], time_precision="hours"):
    """
    Map moments to list of tags listened in the given hour
    (including tag weights)

    "2022-2-23": [("electronic", "98"), ("rock", "59"), ...]
    """
    moment_to_tags_mapping: MomentTags = {}

    for track in trackplays:
        tags = [
            tag for tag in track.all_tags() if type(tag) == list or type(tag) == tuple
        ]

        # Round to minutes. Data points will be by minute
        if track.playback_utc_date:
            moment = track.playback_utc_date.isoformat(timespec=time_precision)

            if tags and moment not in moment_to_tags_mapping:
                moment_to_tags_mapping[moment] = []

            for tagname, tagweight in tags:
                tagname = tagname.lower().strip()
                moment_to_tags_mapping[moment].append((tagname, tagweight))

    return moment_to_tags_mapping


TagsRelevance = Dict[str, int]


def create_tags_dataframe_sorted_by_relevance(moment_to_tags_mapping: MomentTags):
    """
    Returns a dataframe that maps tags to the sum of
    tag weights of all the occurences of the tag

    { "electronic": 4838383.23 }
    """
    tags_relevance: TagsRelevance = {}

    for _, tags in moment_to_tags_mapping.items():
        for tagname, tagweight in tags:
            tags_relevance[tagname] = tags_relevance.get(tagname, 0) + int(tagweight)

    return (
        pd.DataFrame(
            {
                "tag": tags_relevance.keys(),
                "weight": tags_relevance.values(),
            }
        )
        .set_index("tag")
        .sort_values(by=["weight"], ascending=False)
    )


def calculate_tag_relevance_by_moment(
    moment_to_tags_mapping: MomentTags, tagnames: List[str]
):
    """
    For each moment, sum the weights of each tag

    "2022-2-2": [(rock, 2), (rock, 2)] ----> "2022-2-2": {"rock": 4}
    """
    moments_no_duplicate_tags: Dict[str, TagsRelevance] = {}

    for moment, moment_tags in moment_to_tags_mapping.items():
        moment_tag_weights: TagsRelevance = {}

        for tagname, tagweight in moment_tags:
            # Ignore tags that are not relevant enough
            if tagname not in tagnames:
                continue

            if tagname not in moment_tag_weights:
                moment_tag_weights[tagname] = 0

            moment_tag_weights[tagname] += int(tagweight)

        moments_no_duplicate_tags[moment] = moment_tag_weights

    return moments_no_duplicate_tags


def create_moments_to_tags_relevance_df(tagnames, moments_to_tag_relevances_mapping):
    # generate columns for the pandas dataframe
    # First column is the moment time stamp
    # Rest of columns, one column per tag
    #
    # timestamp | tag1 | tag2 | ... | tagN

    timestamps = moments_to_tag_relevances_mapping.keys()

    # Add columns for each tag, all set to 0.
    def zeros():
        return [0] * len(timestamps)

    tag_columns = {tagname: zeros() for tagname in tagnames}
    columns = {"timestamp": [], **tag_columns}

    # Calculate weighted probabilities for each moment
    rowindex = 0
    empty_moments = 0
    for moment, moment_tag_weights in moments_to_tag_relevances_mapping.items():
        # Sum all weights
        total_weight = sum(moment_tag_weights.values())

        if total_weight == 0:
            empty_moments += 1

        columns["timestamp"].append(moment)

        for tagname, tagweight in moment_tag_weights.items():
            columns[tagname][rowindex] = 100 * tagweight / total_weight

        rowindex += 1

    df = pd.DataFrame(columns)

    return df


def convert_tag_weight_tuples_to_str(tags_and_weights: List[Tuple[str, str]]):
    """
    [(rock, 34), (metal, 88)] ----> "rock: 34 metal 88"
    """
    as_strings = []
    for tag, weight in tags_and_weights:
        as_strings.append(f"{tag.strip()}: {weight}")

    return "; ".join(as_strings)


def get_tokens_by_moment(time_precision="hours"):

    """
    Generates a CSV file of tokenized Last.fm tags by moment

    Help: https://huggingface.co/docs/transformers/preprocessing
    """
    trackplays = get_all_lastfm_track_plays()

    moment_to_tags_mapping = map_moments_to_tags(trackplays, time_precision)

    moment_to_tags_string_mapping = {
        moment: convert_tag_weight_tuples_to_str(moment_to_tags_mapping[moment])
        for moment in moment_to_tags_mapping
    }

    batch_sentences = list(moment_to_tags_string_mapping.values())[:10]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded_sentences = tokenizer(batch_sentences, padding=True)
    return tokenizer, encoded_sentences


def get_tags_from_tokens(tokenizer: Tokenizer, encoding: Dict):

    """
    Generates a CSV file of tokenized Last.fm tags by moment
    """

    return tokenizer.decode(encoding["input_ids"])


if __name__ == "__main__":

    tokenizer, encoding = get_tokens_by_moment()

    print(tokenizer.decode(encoding["input_ids"][0]))

    # print(get_tags_from_tokens(tokenizer, encoding))
