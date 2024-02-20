import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";


const VALENCE_INFERENCE_ENDPOINT = process.env.VALENCE_INFERENCE_ENDPOINT;

export async function predictValence(preference: UserPreference) {
    return predictAudioFeature(preference, VALENCE_INFERENCE_ENDPOINT);
}
