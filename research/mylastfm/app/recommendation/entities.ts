export class Preference {

}


export class TextPreference {
    constructor(public text: string) { }
}

export class AudioPreference {

}

export class Track {
    constructor(
        public artist: string,
        public trackName: string,
        public audiofeatures: AudioFeatures
    ) { }
}

export class AudioFeatures {
    constructor(
        public acousticness: number,
        public danceability: number,
        public energy: number,
        public instrumentalness: number,
        public valence: number
    ) { }
}

export class SpotifyAudioFeatures extends AudioFeatures {
    constructor(
        public spotifyID: string,
        acousticness: number,
        danceability: number,
        energy: number,
        instrumentalness: number,
        valence: number
    ) {
        super(acousticness, danceability, energy, instrumentalness, valence);
    }

    static fromDanceability(spotifyID: string, danceability: number) {
        return new SpotifyAudioFeatures(spotifyID, 0, danceability, 0, 0, 0);
    }
}

export class Recommendation {
    constructor(
        public tracks: Array<RecommendedTrack>,
        public features: AudioFeatures
    ) { }
}

class RecommendationDistanceToBestCase {
    constructor(
        public acousticnessDistance: number,
        public danceabilityDistance: number,
        public energyDistance: number,
        public instrumentalnessDistance: number,
        public valenceDistance: number
    ) { }

    total() {
        return this.acousticnessDistance + this.danceabilityDistance +
            this.energyDistance + this.instrumentalnessDistance + this.valenceDistance;
    }
}

export class RecommendedTrack {

    tags: Array<LastfmTag> = [];

    constructor(
        public track: Track,
        public spotifyTrackID: string,
        public distance: RecommendationDistanceToBestCase
    ) { }

    static fromDanceability(
        artist: string,
        trackName: string,
        spotifyTrackID: string,
        danceability: number,
        distance: number
    ) {
        const audioFeatures = SpotifyAudioFeatures.fromDanceability(spotifyTrackID, danceability);
        const track = new Track(artist, trackName, audioFeatures);
        return new RecommendedTrack(
            track,
            spotifyTrackID,
            new RecommendationDistanceToBestCase(0, distance, 0, 0, 0)
        );
    }

    withTags(tags: Array<LastfmTag>) {
        this.tags = tags;
        return this;
    }

}

export class LastfmTag {
    constructor(public tag: string, public weight: number) {}

    toString() {
        return `${this.tag}: ${this.weight}`;
    }
}

