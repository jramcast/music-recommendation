import { Preference, AudioFeatures, TextPreference } from "./entities";
import topTags from "./top-1000-tags.json";


export const SPOTIFY_FEATURES_INFERENCE_ENDPOINT = process.env.SPOTIFY_FEATURES_INFERENCE_ENDPOINT;


export async function predictAudioFeatures(preference: TextPreference) {
    if (SPOTIFY_FEATURES_INFERENCE_ENDPOINT) {
        const payload: ModelServingRequestBody<Array<number>> = {
            "inputs": [
                {
                    "name": "X",
                    "shape": [1, 1],
                    "datatype": "FP32",
                    "data": await buildTagWeightsBasedOnPreferences(preference)
                }
            ]
        };
        const response = await fetch(SPOTIFY_FEATURES_INFERENCE_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload)
        });
        const body = await response.json();
        if (body.code > 0) {
            throw new InferenceServiceError("Error calling inference server", payload, body);
        }
        const danceability = body.outputs[0].data[0];
        return new AudioFeatures(null, danceability, null, null, null);
    }

    return new AudioFeatures(null, null, null, null, null);
}


export interface ModelServingRequestBody<T> {
    inputs: Array<{
        name: string;
        shape: Array<number>;
        datatype: string;
        data: T;
    }>;
}


export class InferenceServiceError extends Error {

    public endpoint = SPOTIFY_FEATURES_INFERENCE_ENDPOINT;
    public payload;
    constructor(message, payload, responseBody) {
        const fullMessage = `${message}: ${responseBody.message}`;
        super(fullMessage);
        this.endpoint = SPOTIFY_FEATURES_INFERENCE_ENDPOINT;
        this.payload = payload;
        console.error(fullMessage);
    }
}


export async function buildTagWeightsBasedOnPreferences(preference: TextPreference) {
    const preferenceWords = preference.text.toLocaleLowerCase().split(" ");

    const tagWeights = topTags
        .map(tag => tag.toLocaleLowerCase())
        .map(tag => {
            for (const [i, preferenceWord] of preferenceWords.entries()) {
                // If the tag includes any of the words included in the input preference,
                // give some weight to this tag. The tag weight depends on the position of the preference word
                // in the preference text.
                if (tag.includes(preferenceWord)) {
                    return Math.floor(100/(1 + i));
                }
            }
            return 0;
        });

    console.log("Tag weights", tagWeights);
    return tagWeights;
}

