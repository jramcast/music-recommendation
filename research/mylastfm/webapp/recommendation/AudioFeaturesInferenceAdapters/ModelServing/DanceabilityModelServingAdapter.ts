import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";


const DANCEABILITY_INFERENCE_ENDPOINT = process.env.DANCEABILITY_INFERENCE_ENDPOINT;

export async function predictDanceability(preference: UserPreference) {
    return predictAudioFeature(preference, DANCEABILITY_INFERENCE_ENDPOINT);
}
