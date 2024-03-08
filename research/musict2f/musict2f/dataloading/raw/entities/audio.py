from dataclasses import dataclass
from typing import Any, Dict, List, Union


# Only in python 3.8
# class SpotifyAnalysis(TypedDict):
#     track: List
#     bars: List
#     beats: List
#     tatums: List
#     sections: List
#     segments: List


@dataclass
class SpotifyAudioFeatures:
    track_name: str
    track_artist: str
    features: List[Dict[str, Union[float, int, str]]]
    analysis: Any

    def has_features(self):
        return len(self.features) and self.features[0] and len(self.features[0].keys())

    def get_features_dict(self, included_features: List[str]):
        features = self.features[0]
        return {
            key: value for key, value in features.items() if key in included_features
        }
