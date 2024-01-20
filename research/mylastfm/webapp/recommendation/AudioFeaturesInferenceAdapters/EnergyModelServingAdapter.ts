import { UserPreference } from "../entities";
import { predictAudioFeature } from "./ModelServingClient";


const ENERGY_INFERENCE_ENDPOINT = process.env.ENERGY_INFERENCE_ENDPOINT;

export async function predictEnergy(preference: UserPreference) {
    return predictAudioFeature(preference, ENERGY_INFERENCE_ENDPOINT);
}
