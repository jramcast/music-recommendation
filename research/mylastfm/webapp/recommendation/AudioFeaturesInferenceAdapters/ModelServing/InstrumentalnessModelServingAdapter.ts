import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";


const INSTRUMENTALNESS_INFERENCE_ENDPOINT = process.env.INSTRUMENTALNESS_INFERENCE_ENDPOINT;

export async function predictInstrumentalness(preference: UserPreference) {
    return predictAudioFeature(preference, INSTRUMENTALNESS_INFERENCE_ENDPOINT);
}
