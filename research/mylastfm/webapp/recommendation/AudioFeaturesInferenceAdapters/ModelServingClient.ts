import { InferenceServiceError, ModelServingRequestBody } from "../AudioFeaturesInference";
import { UserPreference } from "../entities";


export async function predictAudioFeature(preference: UserPreference, endpoint: string) {
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

    const response = await fetch(endpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload)
    });
    const body = await response.json();
    if (body.code > 0) {
        throw new InferenceServiceError("Error calling inference server", endpoint, payload, body);
    }
    const danceability = body.outputs[0].data[0];
    return danceability;
}