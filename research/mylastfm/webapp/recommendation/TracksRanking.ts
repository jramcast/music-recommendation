import mongo = require('mongodb');
import { AudioFeatures, RecommendationDistanceToBestCase, Track } from './entities';
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
        const results = await spotifyAudiofeatures.find<SpotifyTrack>(
            {},
            {
                projection: {
                    features: true,
                    track_name: true,
                    track_artist: true
                }
            }
        )
        .toArray();


        const nearestNeighbors = knn(audioFeatures, results, 10);

        const recommendedTracks = Promise.all(nearestNeighbors.map(async doc => {

            const recommendedTrack = new RecommendedTrack(
                new Track(doc.track_artist, doc.track_name, doc.features[0]),
                doc.features[0].id,

                // Acousticness and Instrumentalness are taken out of the computation because they do not perform well
                new RecommendationDistanceToBestCase(
                    Math.abs(doc.features[0].acousticness - audioFeatures.acousticness),
                    Math.abs(doc.features[0].danceability - audioFeatures.danceability),
                    Math.abs(doc.features[0].energy - audioFeatures.energy),
                    Math.abs(doc.features[0].instrumentalness - audioFeatures.instrumentalness),
                    Math.abs(doc.features[0].valence - audioFeatures.valence)
                )
            );

            const tags = await findLastfmTagsForTrack(recommendedTrack.track);

            return recommendedTrack.withTags(tags);
        }));

        return recommendedTracks;

    } finally {
        await client.close();
    }

}






interface SpotifyTrack {
    features?: Array<{
        id: string;
        danceability: number;
        acousticness: number;
        instrumentalness: number;
        valence: number;
        energy: number;
    }>,
    track_name: string;
    track_artist: string;
}

function euclideanDistance(audioFeatures: AudioFeatures, track: SpotifyTrack): number {
    const dimensions = ["danceability", "acousticness", "instrumentalness", "valence", "energy"];
    let sum = 0;

    if (track.features?.length === 0 || track.features[0] === null) {
        return Number.MAX_VALUE;
    }

    for (const dimension of dimensions) {
        const variability = 0; // -0.2 + (Math.random() * 0.4) Uncomment to add randomness
        const diff = variability + audioFeatures[dimension] - track.features[0][dimension];
        sum += diff * diff;
    }

    return Math.sqrt(sum);
}

function knn(audioFeatures: AudioFeatures, dataset: SpotifyTrack[], k: number): SpotifyTrack[] {
    const distances: { song: SpotifyTrack, distance: number }[] = [];

    for (const song of dataset) {
        const distance = euclideanDistance(audioFeatures, song);
        distances.push({ song, distance });
    }

    distances.sort((a, b) => a.distance - b.distance);

    const topK = distances.slice(0, k);

    return topK.map(item => item.song);
}









/**
 * Get a list of tracks closest to the provided audio features
 * @param {AudioFeatures} audioFeatures
 */
export async function fetchRecommendationsOld(audioFeatures: AudioFeatures) {

    const client = new MongoClient(MONGODB_ADDRESS);

    try {
        await client.connect();
        const db = client.db("mgr");

        const spotifyAudiofeatures = db.collection("spotify_audiofeatures");
        const resultsBelow = await spotifyAudiofeatures.find(
            { "features.danceability": { "$lte": audioFeatures.danceability } },
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
            { "features.danceability": { "$gt": audioFeatures.danceability } },
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