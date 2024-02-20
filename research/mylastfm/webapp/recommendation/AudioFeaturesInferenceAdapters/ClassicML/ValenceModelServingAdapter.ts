import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";

export async function predictValence(preference: UserPreference) {
    return predictAudioFeature(preference, process.env.CLASSICML_INFERENCE_ENDPOINT, "valence");
}
