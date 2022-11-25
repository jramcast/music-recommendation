from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass
from .artist import Artist


@dataclass
class Track:
    artist: Artist
    name: str
    tags: List[List[str]]
    artist_tags: List[List[str]]
    loved: bool
    user_playcount: int
    total_playcount: int
    playback_utc_date: Optional[datetime]
    mbid: Optional[str] = None
    _id: Optional[str] = None

    def all_tags(self):
        """
        Get track tags +  artist tags
        """
        return self.tags + self.artist_tags

    def __repr__(self) -> str:
        return f"{self.artist.name} - '{self.name}' at '{self.playback_utc_date}'"
