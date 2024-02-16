import { predictAcousticness } from "./AudioFeaturesInferenceAdapters/AcousticnessModelServingAdapter";
import { predictDanceability } from "./AudioFeaturesInferenceAdapters/DanceabilityModelServingAdapter";
import { predictEnergy } from "./AudioFeaturesInferenceAdapters/EnergyModelServingAdapter";
import { predictInstrumentalness } from "./AudioFeaturesInferenceAdapters/InstrumentalnessModelServingAdapter";
import { predictValence } from "./AudioFeaturesInferenceAdapters/ValenceModelServingAdapter";

import { predictAcousticness as predictAcousticnessGPT2 } from "./AudioFeaturesInferenceAdapters/GTP2/AcousticnessModelServingAdapter";
import { predictDanceability as predictDanceabilityGPT2 } from "./AudioFeaturesInferenceAdapters/GTP2/DanceabilityModelServingAdapter";
import { predictEnergy as predictEnergyGPT2 } from "./AudioFeaturesInferenceAdapters/GTP2/EnergyModelServingAdapter";
import { predictInstrumentalness as predictInstrumentalnessGPT2 } from "./AudioFeaturesInferenceAdapters/GTP2/InstrumentalnessModelServingAdapter";
import { predictValence as predictValenceGPT2 } from "./AudioFeaturesInferenceAdapters/GTP2/ValenceModelServingAdapter";


import { AudioFeatures, UserPreference } from "./entities";


export async function predictAudioFeatures(preference: UserPreference) {

    const [ danceability, acousticness, energy, valence, instrumentalness ] = await Promise.all([
        predictDanceabilityGPT2(preference),
        predictAcousticnessGPT2(preference),
        predictEnergyGPT2(preference),
        predictValenceGPT2(preference),
        predictInstrumentalnessGPT2(preference)
    ]);
    // const [ danceability, acousticness, energy, valence, instrumentalness ] =  [
    //     0.428,
    //     0.01,
    //     0.933,
    //     0.609,
    //     0.000134
    // ]

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

    public endpoint: string;
    public payload;

    constructor(message: string, endpoint: string, payload, responseBody) {
        const fullMessage = `${message}: ${responseBody.message}`;
        super(fullMessage);
        this.endpoint = endpoint;
        this.payload = payload;
        console.error(fullMessage);
    }
}

