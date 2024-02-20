import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";


const ACOUSTICNESS_INFERENCE_ENDPOINT = process.env.ACOUSTICNESS_INFERENCE_ENDPOINT;

export async function predictAcousticness(preference: UserPreference) {
    return predictAudioFeature(preference, ACOUSTICNESS_INFERENCE_ENDPOINT);
}
