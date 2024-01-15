import { ModelServingRequestBody, InferenceServiceError } from "../AudioFeaturesInference";
import { AudioFeatures, UserPreference } from "../entities";


const DANCEABILITY_INFERENCE_ENDPOINT = process.env.DANCEABILITY_INFERENCE_ENDPOINT;

export async function predictDanceability(preference: UserPreference) {

    const payload: ModelServingRequestBody<Array<number>> = {
        "inputs": [
            {
                "name": "X",
                "shape": [1, 1],
                "datatype": "FP32",
                "data": preference.asTop1000TagWeights()
            }
        ]
    };
    const response = await fetch(DANCEABILITY_INFERENCE_ENDPOINT, {
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
    return danceability;
}