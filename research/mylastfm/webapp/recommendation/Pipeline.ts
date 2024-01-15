import { Recommendation } from "./entities";
import { predictAudioFeatures } from "./AudioFeaturesInference";
import { fetchRecommendations } from "./TracksRanking";


/**
 * Recommends a list of tracks based on preferences
 * @param {Preference} preference The text or audio-based preference
 *
 * @returns {Recommendation}
 */
export async function recommend(preference) {
    let audioFeatures = await predictAudioFeatures(preference);
    let tracks = await fetchRecommendations(audioFeatures);
    return new Recommendation(tracks, audioFeatures);
}
