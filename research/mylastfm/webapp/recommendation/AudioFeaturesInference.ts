import { predictAcousticness } from "./AudioFeaturesInferenceAdapters/ModelServing/AcousticnessModelServingAdapter";
import { predictDanceability } from "./AudioFeaturesInferenceAdapters/ModelServing/DanceabilityModelServingAdapter";
import { predictEnergy } from "./AudioFeaturesInferenceAdapters/ModelServing/EnergyModelServingAdapter";
import { predictInstrumentalness } from "./AudioFeaturesInferenceAdapters/ModelServing/InstrumentalnessModelServingAdapter";
import { predictValence } from "./AudioFeaturesInferenceAdapters/ModelServing/ValenceModelServingAdapter";

import { predictAcousticness as predictAcousticnessClassicML } from "./AudioFeaturesInferenceAdapters/ClassicML/AcousticnessModelServingAdapter";
import { predictDanceability as predictDanceabilityClassicML } from "./AudioFeaturesInferenceAdapters/ClassicML/DanceabilityModelServingAdapter";
import { predictEnergy as predictEnergyClassicML } from "./AudioFeaturesInferenceAdapters/ClassicML/EnergyModelServingAdapter";
import { predictInstrumentalness as predictInstrumentalnessClassicML } from "./AudioFeaturesInferenceAdapters/ClassicML/InstrumentalnessModelServingAdapter";
import { predictValence as predictValenceClassicML } from "./AudioFeaturesInferenceAdapters/ClassicML/ValenceModelServingAdapter";


import { AudioFeatures, UserPreference } from "./entities";


export async function predictAudioFeatures(preference: UserPreference) {

    const [ danceability, acousticness, energy, valence, instrumentalness ] = await Promise.all([
        predictDanceabilityClassicML(preference),
        predictAcousticnessClassicML(preference),
        predictEnergyClassicML(preference),
        predictValenceClassicML(preference),
        predictInstrumentalnessClassicML(preference)
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

