import fs = require("fs");
import path = require("path");
import { Recommendation, Track, Preference, AudioFeatures } from "./entities";
import { getTracksRanking } from "../tracksRanking/service";
import topTags from "./top-1000-tags.json";

module.exports = {
    recommend
}


const SPOTIFY_FEATURES_INFERENCE_ENDPOINT = process.env.SPOTIFY_FEATURES_INFERENCE_ENDPOINT;


/**
 * Recommends a list of tracks based on preferences
 * @param {Preference} preference The text or audio-based preference
 *
 * @returns {Recommendation}
 */
async function recommend(preference) {
    let features = await fetchFeaturePreditions(preference);
    let tracks = await fetchRecommendations(features);
    return new Recommendation(tracks, features);
}



interface ModelServingRequestBody<T> {
    inputs: Array<{
        name: string;
        shape: Array<number>
        datatype: string;
        data: T;
    }>
}

async function fetchFeaturePreditions(preference: Preference) {
    if (SPOTIFY_FEATURES_INFERENCE_ENDPOINT) {
        const payload: ModelServingRequestBody<Array<number>> = {
            "inputs": [
                {
                    "name": "X",
                    "shape": [1,1],
                    "datatype": "FP32",
                    "data": await buildDataBasedOnPreferences(preference)
                }
            ]
        }
        const response = await fetch(SPOTIFY_FEATURES_INFERENCE_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type":"application/json",
            },
            body: JSON.stringify(payload)
        });
        const body = await response.json();
        if (body.code > 0) {
            throw new InferenceServiceError("Error calling inference server", payload, body)
        }
        const danceability = body.outputs[0].data[0];
        return new AudioFeatures(null, danceability, null, null, null);
    }

    return new AudioFeatures(null, null, null, null, null);
}

class InferenceServiceError extends Error {

    public endpoint = SPOTIFY_FEATURES_INFERENCE_ENDPOINT;
    public payload
    constructor(message, payload, responseBody) {
        const fullMessage = `${message}: ${responseBody.message}`;
        super(fullMessage);
        this.endpoint = SPOTIFY_FEATURES_INFERENCE_ENDPOINT;
        this.payload = payload;
        console.error(fullMessage);
    }
}

/**
 *
 * @param {AudioFeatures} audioFeatures
 * @returns
 */
async function fetchRecommendations(audioFeatures) {
    const trackInfos = await getTracksRanking(audioFeatures);
    const tracks = trackInfos.map(toTrack);
    return tracks;
}


/**
 *
 * @param {TextPreference} preference
 */
async function buildDataBasedOnPreferences(preference) {
    const preferenceWords = preference.text.split(" ");

    return topTags.map(tag => preferenceWords.includes(topTags) ? 1 : 0)
}


function toTrack(trackInfo) {
    return new Track(trackInfo.track_artist, trackInfo.track_name, trackInfo.features);
}

