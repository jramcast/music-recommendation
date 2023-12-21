import mongo = require('mongodb');
import { Track, AudioFeatures } from '../recommendation/entities';
const { MongoClient, ServerApiVersion } = mongo;



module.exports = {
    getTracksRanking
}

const MONGODB_ADDRESS = process.env.MONGODB_ADDRESS;



class DistanceToBestCase {
    constructor(
        public acousticnessDistance: number,
        public danceabilityDistance: number,
        public energyDistance: number,
        public instrumentalnessDistance: number,
        public valenceDistance: number
    ) { }
}

class RecommendedTrack {
    constructor(public track: Track, public distance: DistanceToBestCase ) {}

    static fromDanceability(
        artist: string,
        trackName: string,
        danceability: number,
        distance: number
    ) {
        const audioFeatures = AudioFeatures.fromDanceability(danceability);
        const track = new Track(artist, trackName, audioFeatures);
        return new RecommendedTrack(track, new DistanceToBestCase(NaN, distance, NaN, NaN, NaN))
    }

}


/**
 * Get a list of tracks closest to the provided audio features
 * @param {AudioFeatures} audioFeatures
 */
export async function getTracksRanking(audioFeatures) {

    const client = new MongoClient(MONGODB_ADDRESS, {
        serverApi: {
            version: ServerApiVersion.v1,
            strict: true,
            deprecationErrors: true,
        }
    });

    try {
        await client.connect();
        const db = client.db("mgr");

        const spotifyAudiofeatures = db.collection("spotify_audiofeatures");
        const resultsBelow = await spotifyAudiofeatures.find(
                {"features.danceability": {"$lte": audioFeatures.danceability}},
                {
                    projection: {
                        features: true,
                        track_name: true,
                        track_artist: true
                    }
                }
            )
            .sort("features.danceability", -1)
            .limit(10)
            .toArray();

        const resultsAbove = await spotifyAudiofeatures.find(
                {"features.danceability": {"$gt": audioFeatures.danceability}},
                {
                    projection: {
                        features: true,
                        track_name: true,
                        track_artist: true
                    }
                }
            )
            .sort("features.danceability", 1)
            .limit(10)
            .toArray();

        const results = [...resultsBelow, ...resultsAbove].map(doc => RecommendedTrack.fromDanceability(
            doc.track_artist,
            doc.track_name,
            doc.features[0].danceability,
            Math.abs(doc.features[0].danceability - audioFeatures.danceability)
        ));

        return results;

    } finally {
        await client.close();
    }

}