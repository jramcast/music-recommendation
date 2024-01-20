import topTags from "./top-1000-tags.json";

export abstract class UserPreference {
    abstract asTop1000TagWeights();
    abstract asText();
}


export class UserPreferenceAsText extends UserPreference {
    constructor(public text: string) {
        super();
    }

    asText() {
        return this.text;
    }

    asTop1000TagWeights() {

        const stopwords = [",",".","", "i", "me", "my", "myself", "we", "our",
        "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now"];

        const preferenceWords = this.text.toLocaleLowerCase()
            .split(/\s|\b/)
            .map(word => word.trim())
            .filter(word => !stopwords.includes(word));

        console.log(preferenceWords);

        const tagWeights = topTags
            .map(tag => tag.toLocaleLowerCase())
            .map(tag => {
                for (const [i, preferenceWord] of preferenceWords.entries()) {
                    // If the tag includes any of the words included in the input preference,
                    // give some weight to this tag. The tag weight depends on the position of the preference word
                    // in the preference text.
                    if (tag.includes(preferenceWord)) {
                        return Math.floor(100/(1 + Math.log10(i + 1)));
                    }
                }
                return 0;
            });

        console.log(tagWeights);
        console.log("---");

        // var randomArray = [];
        // for (var i = 0; i < topTags.length; i++) {
        //     randomArray.push(Math.floor(Math.random() * 101));
        // }
        // return randomArray;

        return tagWeights;
    }
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

export class RecommendationDistanceToBestCase {
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

