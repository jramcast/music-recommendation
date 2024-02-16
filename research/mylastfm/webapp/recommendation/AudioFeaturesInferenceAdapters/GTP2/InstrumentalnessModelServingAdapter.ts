import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";


export async function predictInstrumentalness(preference: UserPreference) {
    return predictAudioFeature(preference, process.env.GPT2_INFERENCE_ENDPOINT, "instrumentalness");
}
