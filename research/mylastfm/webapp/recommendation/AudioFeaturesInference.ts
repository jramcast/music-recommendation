import { predictAcousticness } from "./AudioFeaturesInferenceAdapters/AcousticnessModelServingAdapter";
import { predictDanceability } from "./AudioFeaturesInferenceAdapters/DanceabilityModelServingAdapter";
import { predictEnergy } from "./AudioFeaturesInferenceAdapters/EnergyModelServingAdapter";
import { predictInstrumentalness } from "./AudioFeaturesInferenceAdapters/InstrumentalnessModelServingAdapter";
import { predictValence } from "./AudioFeaturesInferenceAdapters/ValenceModelServingAdapter";
import { AudioFeatures, UserPreference } from "./entities";


export async function predictAudioFeatures(preference: UserPreference) {

    const [ danceability, acousticness, energy, valence, instrumentalness ] = await Promise.all([
        predictDanceability(preference),
        predictAcousticness(preference),
        predictEnergy(preference),
        predictValence(preference),
        predictInstrumentalness(preference)
    ]);

    return new AudioFeatures(acousticness, danceability, energy, instrumentalness, valence);
}


export interface ModelServingRequestBody<T> {
    inputs: Array<{
        name: string;
        shape: Array<number>;
        datatype: string;
        data: T;
    }>;
}


export class InferenceServiceError extends Error {

    public endpoint = "todo";
    public payload;
    constructor(message, payload, responseBody) {
        const fullMessage = `${message}: ${responseBody.message}`;
        super(fullMessage);
        this.endpoint = "todo";
        this.payload = payload;
        console.error(fullMessage);
    }
}

