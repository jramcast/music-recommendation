"""
Last.fm data preprocessing before training.

This scripts reads the raw Last.fm from a MongoDB and creates a CSV
"""

from operator import itemgetter
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Iterable, List, Union

import pandas as pd
from pymongo import MongoClient
from tokenizers import Tokenizer, Encoding


from dataloading.raw.entities.track import Track
from dataloading.raw.lastfm import MongoDBTrackPlaysRepository
from . import spotify

# A function that represents tags|weights as a string
Stringifier = Callable[[Union[List[Tuple[str, str]], List[List[str]]]], str]


def create_tag_tokens_by_moment_csv(
    csv_filename: Path,
    tokenizer: Tokenizer,
    stringifier: Stringifier,
    time_precision="hours",
):

    """
    Generates a CSV file of tokenized Last.fm tags by moment

    Help: https://huggingface.co/docs/transformers/preprocessing
    """
    trackplays = get_all_lastfm_track_plays()

    moment_to_tags_mapping = map_moments_to_tags(trackplays, time_precision)

    # Each moment maps to a str
    moment_to_tags_string_mapping = {
        moment: stringifier(moment_to_tags_mapping[moment])
        for moment in moment_to_tags_mapping
    }

    moment_to_token_ids_mapping = convert_moment_tag_strings_to_tokens(
        moment_to_tags_string_mapping, tokenizer
    )

    # Orient="index" to use dict keys as rows
    df = pd.DataFrame.from_dict(moment_to_token_ids_mapping, orient="index")
    df.index.rename("timestamp", inplace=True)

    df.to_csv(csv_filename, index=True)

    return df


def create_texts_by_moment_csv(
    csv_filename: Path,
    stringifier: Stringifier,
    time_precision="hours",
):

    """
    Generates a CSV file of concatenated Last.fm tags by moment

    Help: https://huggingface.co/docs/transformers/preprocessing
    """
    trackplays = get_all_lastfm_track_plays()

    moment_to_tags_mapping = map_moments_to_tags(trackplays, time_precision)

    # Each moment maps to a str
    moment_to_tags_string_mapping = {
        moment: stringifier(moment_to_tags_mapping[moment])
        for moment in moment_to_tags_mapping
    }

    # Orient="index" to use dict keys as rows
    df = pd.DataFrame.from_dict(moment_to_tags_string_mapping, orient="index")
    df.index.rename("timestamp", inplace=True)
    df.rename(columns={0: "text"}, inplace=True)

    df.to_csv(csv_filename, index=True)

    return df


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


def get_tag_probs_by_track(tags_limit=999999999):
    """
    Returns a dict of Last.fm tag relevances by track

    { "artist - track": { "rock": 100, "pop": 100 }}
    """
    trackplays = get_all_lastfm_track_plays()

    moment_to_tags_mapping = map_moments_to_tags(trackplays, "hours")

    tag_relevances = create_tags_dataframe_sorted_by_relevance(moment_to_tags_mapping)

    # Limit number of tags to include
    tag_relevances = tag_relevances[:tags_limit]

    # Get the list of tags
    tagnames = tag_relevances.index.tolist()

    # Create tags by song mapping
    tracks_to_tag_relevances_mapping = get_all_last_fm_tags_as_dicts_by_track(tagnames)

    return tracks_to_tag_relevances_mapping


def get_all_lastfm_track_plays():
    client = MongoClient()
    db = client.mgr
    repo = MongoDBTrackPlaysRepository(db.lastfm_playedtracks)
    trackplays = repo.all()
    return trackplays


def get_all_last_fm_tags_as_dicts_by_track(tags_included: Optional[List[str]] = None):
    """
    Returns a dict that maps track keys (artist - track)
    to a dictionary a tag relevances:

      {"artist - name": {"rock": 23, "electronic" 1}}

    """
    lastfm_tags_by_track = {}
    track_plays = get_all_lastfm_track_plays()
    for trackplay in track_plays:
        tags = [
            tag
            for tag in trackplay.all_tags()
            if type(tag) == list or type(tag) == tuple
        ]
        key = spotify.get_trackanalysis_key(trackplay.artist.name, trackplay.name)

        tags_dict = {t: 0 for t in tags_included or []}

        for tag_name, tag_weight in tags:
            if not tags_included or tag_name in tags_included:
                tags_dict[tag_name] = int(tag_weight)

        lastfm_tags_by_track[key] = tags_dict

    return lastfm_tags_by_track


def get_all_last_fm_tags_as_text_by_song(stringifier: Stringifier):
    """
    Returns a dict that maps track keys (artist - track)
    to track tags, represented as a string

        {"artist - name": "rock, electronic ..."
    """
    lastfm_tags_by_track = {}
    track_plays = get_all_lastfm_track_plays()
    for trackplay in track_plays:
        tags = [
            tag
            for tag in trackplay.all_tags()
            if type(tag) == list or type(tag) == tuple
        ]
        key = spotify.get_trackanalysis_key(trackplay.artist.name, trackplay.name)
        lastfm_tags_by_track[key] = stringifier(tags)

    return lastfm_tags_by_track


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


def convert_moment_tag_strings_to_tokens(
    moment_to_string_mapping: Dict[str, str], tokenizer: Tokenizer
):
    moment_to_token_ids_mapping: Dict[str, List[int]] = {}

    for moment in moment_to_string_mapping:
        text = moment_to_string_mapping[moment]
        encoding: Encoding = tokenizer.encode(text)
        moment_to_token_ids_mapping[moment] = encoding.ids

    return moment_to_token_ids_mapping


def init_tokenizer(token_limit: int):
    tokenizer: Tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    max_length = token_limit
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length)
    return tokenizer


def tag_stringifier_include_weight(tags_and_weights: List[Tuple[str, str]]):
    """
    [(rock, 34), (metal, 88)] ----> "'rock' 34, 'metal' 88"
    """
    as_strings = []
    for tag, weight in tags_and_weights:
        as_strings.append(f"'{tag.strip()}' {weight}")

    return ", ".join(as_strings)


def tag_stringifier_repeat_tag(tags_and_weights: List[Tuple[str, str]]):
    """
    [(rock, 2), (metal, 4)] ----> "'rock rock, metal metal metal metal"
    """
    as_strings = []
    for tag, weight in tags_and_weights:
        as_strings.append(" ".join([tag.strip()] * int(weight)))

    return ", ".join(as_strings)


def tag_stringifier_weight_as_order(tags_and_weights: List[Tuple[str, str]]):
    """
    [(rock, 2), (metal, 4)] ----> "metal, rock"
    """
    as_strings = []
    tags_and_weights_sorted = sorted(tags_and_weights, key=itemgetter(1), reverse=True)

    for tag, weight in tags_and_weights:
        as_strings.append(" ".join([tag.strip()] * int(weight)))

    return ", ".join([tag for tag, weight in tags_and_weights_sorted])


def get_tags_from_token_ids(tokenizer: Tokenizer, ids: List[int]):

    """
    Generates a CSV file of tokenized Last.fm tags by moment
    """
    return tokenizer.decode(ids)
