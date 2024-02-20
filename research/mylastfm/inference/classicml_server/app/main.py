import os
import re
import json
import math
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import pandas as pd
from joblib import load
import numpy as np
import xgboost
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches


MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).parent / "_models"))


def load_xgboost_model(filepath: str):
    m = xgboost.XGBRegressor(n_estimators=200)
    m.load_model(filepath)
    return m


features = ["acousticness", "danceability", "energy", "instrumentalness", "valence"]
scalers = {f: load(MODELS_DIR / f"scaler_{f}.joblib") for f in features}
models = {
    "bayes": {
        f: load(MODELS_DIR / f"{f}-bayes-1000-probs-by_track.json") for f in features
    },
    "xgboost": {
        f: load_xgboost_model(MODELS_DIR / f"{f}-xgboost-1000-probs-by_track.json")
        for f in features
    },
}


with open(Path(__file__).parent / "top-1000-tags.json") as f:
    columns = json.loads(f.read())


with open(Path(__file__).parent / "stopwords.json") as f:
    stopwords = json.loads(f.read())


def build_input(text: str):
    preference_words = [word for word in re.split(r"\s+|[^\w\s]", text.lower())]
    preference_words = [
        word
        for word in preference_words
        if word and len(word) > 1 and word not in stopwords
    ]

    def calculate_close_matches_similarity(text: str, concept: str):
        # Create TfidfVectorizer instance
        vectorizer = TfidfVectorizer()

        # Fit the text
        vectorized_text = vectorizer.fit_transform([text])

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Find similar terms using difflib
        matches = get_close_matches(concept, feature_names, n=1, cutoff=0.9)

        if not matches:
            return 0

        matched_concept = matches[0]

        # Transform the matched concept
        concept_vector = vectorizer.transform([matched_concept])

        # Calculate cosine similarity between the text and concept vector
        cosine_sim = np.dot(vectorized_text.toarray(), concept_vector.toarray().T)

        return cosine_sim[0][0]

    def calculate_tfidf_similarity(text: str, column: str, position_weight=0.03):
        # Create TfidfVectorizer instance
        vectorizer = TfidfVectorizer()

        # Fit the text
        vectorized_text = vectorizer.fit_transform([text])

        # Transform the concept
        concept_vector = vectorizer.transform([column])

        # Get feature names and IDF values
        feature_names = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_

        # Get term positions in the text
        term_positions = [
            text.index(term) if term in text else -1 for term in feature_names
        ]

        # Calculate TF-IDF weighted by term positions
        weighted_tfidf = []
        for term, idf, position in zip(feature_names, idf_values, term_positions):
            if position != -1:
                tfidf = idf * (
                    1 / (1 + position_weight * position)
                )  # Weight by term position
                weighted_tfidf.append(tfidf)
            else:
                weighted_tfidf.append(0)

        # Normalize weighted TF-IDF values
        normalized_weighted_tfidf = np.array(weighted_tfidf) / np.linalg.norm(
            weighted_tfidf
        )

        # Calculate cosine similarity between the normalized weighted TF-IDF values
        # and the concept vector
        cosine_sim = np.dot(
            normalized_weighted_tfidf, concept_vector.toarray().flatten()
        )

        return cosine_sim

    def calculate_tag_weight(column: str):

        # https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python#need-for-contextual-embeddings
        text_lower, column_lower = text.lower(), column.lower()
        return (
            5 * int(column_lower in text_lower)
            + 3 * calculate_tfidf_similarity(text_lower, column_lower)
            + 2 * calculate_close_matches_similarity(text_lower, column_lower)
        ) / 10

    weights = np.array([calculate_tag_weight(column) for column in columns])
    normalized_weights = np.round(
        100 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    )

    df = pd.DataFrame([normalized_weights], columns=columns, dtype="int")

    nonzero_values = df.iloc[0][df.iloc[0] != 0]
    for column, value in nonzero_values.items():
        print(f"Tag: {column}={value}")

    return df


def inverse_feature_transform(value, feature):
    # Inverse transform to get back the normalized target variable data
    normal_data = scalers[feature].inverse_transform(np.array(value).reshape(-1, 1))

    # Reverse the log transformation
    value = np.exp(normal_data) - 0.0001

    # Inverse of the normalization
    value = value / (1 + value)

    return value[0][0]


api = FastAPI()  # uvicorn app.main:api --reload


class InferenceRequestBody(BaseModel):
    text: str


@api.get("/")
def read_root():
    return {
        "INFO": 'Send a POST request to /predict/{feature} and send a JSON body as { "text": "..."} '
    }


@api.post("/predict/{model}/{feature}")
def predict(model: str, feature: str, body: InferenceRequestBody):
    if model not in models:
        raise HTTPException(status_code=404, detail=f"The {model} model does not exist")

    if feature not in models[model]:
        raise HTTPException(
            status_code=404, detail=f"The {feature} feature does not exist"
        )

    return {feature: predict_audio_feature(model, feature, body.text)}


def predict_audio_feature(model: str, feature: str, text: str):
    input = build_input(text)
    value = models[model][feature].predict(input)
    v = inverse_feature_transform(value, feature)
    return float(v)
