from typing import Iterable, Optional

import pymongo
from pymongo.collection import Collection

from dataclasses import asdict
from dataloading.raw.entities.track import Track
from dataloading.raw.entities.audio import SpotifyAudioFeatures


class MongoDBSpotifyAudioFeaturesRepository:

    collection: Collection

    def __init__(self, collection: Collection):
        self.collection = collection
        self.collection.create_index(
            [("track_name", pymongo.ASCENDING), ("track_artist", pymongo.ASCENDING)],
            unique=True,
        )

    def load(self, track: Track) -> Optional[SpotifyAudioFeatures]:
        return self.collection.find_one(
            {"track_name": track.name, "track_artist": track.artist.name}
        )

    def all(self) -> Iterable[SpotifyAudioFeatures]:
        for doc in self.collection.find():
            yield self._as_audio_features(doc)

    def all_features(self) -> Iterable[SpotifyAudioFeatures]:
        for doc in self.collection.find(
            {},
            {"features": True, "track_name": True, "track_artist": True}
        ):
            yield self._as_audio_features(doc)

    def save(self, features: SpotifyAudioFeatures):
        self.collection.insert_one(asdict(features))

    def _as_audio_features(self, doc) -> SpotifyAudioFeatures:

        return SpotifyAudioFeatures(
            doc["track_name"],
            doc["track_artist"],
            doc["features"],
            doc.get("analysis", None),
        )
