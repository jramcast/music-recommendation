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

    static fromDanceability(danceability: number) {
        return new AudioFeatures(-1, danceability, -1, -1, -1);
    }
}

export class Recommendation {
    constructor(
        public tracks: Array<Track>,
        public features: AudioFeatures
    ) { }
}
