import os

from pathlib import Path
from pydantic import BaseModel
from transformers import pipeline
from fastapi import FastAPI, HTTPException


MODELS_DIR = os.getenv("MODELS_DIR", "app/_models")


api = FastAPI()  # uvicorn main:api --reload
pipelines = {
    "danceability": pipeline(
        "text-classification",
        model=Path(MODELS_DIR) / "danceability-gpt2-_tag_texts-from_tag_order-by_track",
    ),
    "acousticness": pipeline(
        "text-classification",
        model=Path(MODELS_DIR) / "acousticness-gpt2-_tag_texts-from_tag_order-by_track",
    ),
    "energy": pipeline(
        "text-classification",
        model=Path(MODELS_DIR) / "energy-gpt2-_tag_texts-from_tag_order-by_track",
    ),
    "valence": pipeline(
        "text-classification",
        model=Path(MODELS_DIR) / "valence-gpt2-_tag_texts-from_tag_order-by_track",
    ),
    "instrumentalness": pipeline(
        "text-classification",
        model=Path(MODELS_DIR) / "instrumentalness-gpt2-_tag_texts-from_tag_order-by_track",
    ),
}


class InferenceRequestBody(BaseModel):
    text: str


@api.get("/")
def read_root():
    return {
        "INFO": 'Send a POST request to /predict/{feature} and send a JSON body as { "text": "..."} '
    }


@api.post("/predict/{feature}")
def predict(feature, body: InferenceRequestBody):
    if feature not in pipelines:
        raise HTTPException(
            status_code=404, detail=f"The {feature} feature does not exist"
        )

    return {feature: predict_audio_feature(feature, body.text)}


def predict_audio_feature(feature: str, text: str):
    result = pipelines[feature](text)
    return result[0]["score"]
