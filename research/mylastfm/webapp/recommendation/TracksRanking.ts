import mongo = require('mongodb');
import { AudioFeatures } from './entities';
import { RecommendedTrack } from './entities';
import { findLastfmTagsForTrack } from './LastfmTags';
const { MongoClient, ServerApiVersion } = mongo;


const MONGODB_ADDRESS = process.env.MONGODB_ADDRESS;


/**
 * Get a list of tracks closest to the provided audio features
 * @param {AudioFeatures} audioFeatures
 */
export async function fetchRecommendations(audioFeatures: AudioFeatures) {

    const client = new MongoClient(MONGODB_ADDRESS);

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



        const results = await Promise.all([...resultsBelow, ...resultsAbove].map(async doc => {

            const recommendedTrack = RecommendedTrack.fromDanceability(
                doc.track_artist,
                doc.track_name,
                doc.features[0].id,
                doc.features[0].danceability,
                Math.abs(doc.features[0].danceability - audioFeatures.danceability)
            );

            const tags = await findLastfmTagsForTrack(recommendedTrack.track);

            return recommendedTrack.withTags(tags);
        }));

        function byDistance(a: RecommendedTrack, b: RecommendedTrack) {
            return a.distance.total() - b.distance.total();
        }

        results.sort(byDistance);

        return results;

    } finally {
        await client.close();
    }

}