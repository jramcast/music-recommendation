import { ModelServingRequestBody, InferenceServiceError } from "../AudioFeaturesInference";
import { AudioFeatures, UserPreference } from "../entities";


const DANCEABILITY_INFERENCE_ENDPOINT = process.env.DANCEABILITY_INFERENCE_ENDPOINT;

export async function predictAcousticness(preference: UserPreference) {
    return 0;
}