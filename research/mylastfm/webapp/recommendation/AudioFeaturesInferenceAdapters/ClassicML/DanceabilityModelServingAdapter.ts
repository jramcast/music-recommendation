import { UserPreference } from "../../entities";
import { predictAudioFeature } from "./ModelServingClient";


export async function predictDanceability(preference: UserPreference) {
    return predictAudioFeature(preference, process.env.CLASSICML_INFERENCE_ENDPOINT, "danceability");
}
