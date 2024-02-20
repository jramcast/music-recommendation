import { InferenceServiceError } from "../../AudioFeaturesInference";
import { UserPreference, UserPreferenceAsText } from "../../entities";


export async function predictAudioFeature(preference: UserPreference, endpoint: string, feature: string) {

    const payload = preference;
    const url = endpoint.replace("<feature>", feature)
    console.log(url);
    console.log(JSON.stringify(payload))
    const response = await fetch(url, {
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

    return body[feature];
}