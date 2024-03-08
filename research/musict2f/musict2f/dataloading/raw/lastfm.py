from dataclasses import asdict
from typing import Iterable, Optional

import pymongo
import pymongo.collection

from dataloading.raw.entities.audio import SpotifyAudioFeatures
from dataloading.raw.entities.track import Track
from dataloading.raw.entities.artist import Artist


class MongoDBTrackPlaysRepository:

    collection: pymongo.collection.Collection

    def __init__(self, collection: pymongo.collection.Collection):
        self.collection = collection
        self.collection.create_index(
            [
                ("playback_utc_date", pymongo.ASCENDING),
                ("name", pymongo.ASCENDING),
                ("artist.name", pymongo.ASCENDING)
            ],
            unique=True
        )

    def all(self, recent_first=True):
        for doc in self.all_raw(recent_first):
            yield self._as_track(doc)

    def all_raw(self, recentFirst=False):
        return (self.collection
                    .find()
                    .sort("playback_utc_date", 1 if not recentFirst else -1))

    def save(self, track: Track):
        if track._id:
            self.collection.replace_one({"_id": track._id}, asdict(track))
        else:
            self.collection.insert_one(asdict(track))

    def _as_track(self, doc) -> Track:

        artist = Artist(
            doc["artist"]["name"],
            doc["artist"]["mbid"]
        )

        return Track(
            artist,
            doc["name"],
            doc["tags"],
            doc.get("artist_tags", []),
            doc["loved"],
            doc["user_playcount"],
            doc["total_playcount"],
            doc["playback_utc_date"],
            doc["mbid"],
            doc["_id"]
        )


class MongoDBSpotifyAudioFeaturesRepository:

    collection: pymongo.collection.Collection

    def __init__(self, collection: pymongo.collection.Collection):
        self.collection = collection
        self.collection.create_index(
            [
                ("track_name", pymongo.ASCENDING),
                ("track_artist", pymongo.ASCENDING)
            ],
            unique=True
        )

    def load(self, track: Track) -> Optional[SpotifyAudioFeatures]:
        return self.collection.find_one({
            "track_name": track.name,
            "track_artist": track.artist.name
        })

    def all(self) -> Iterable[SpotifyAudioFeatures]:
        for doc in self.collection.find():
            yield self._as_audio_features(doc)

    def save(self, features: SpotifyAudioFeatures):
        self.collection.insert_one(asdict(features))

    def _as_audio_features(self, doc) -> SpotifyAudioFeatures:

        return SpotifyAudioFeatures(
            doc["track_name"],
            doc["track_artist"],
            doc["features"],
            doc["analysis"],
        )
