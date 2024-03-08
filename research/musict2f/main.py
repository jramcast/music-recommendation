import os
import torch
import logging
from pathlib import Path
from transformers import MistralForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from musict2f.training import merlinite7b

logging.info(f"Cuda: {torch.cuda.is_available()}")

DEFAULT_DATA_DIR = Path(__file__).parent.joinpath("../../data/jaime_lastfm")
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
MODELS_SAVE_DIR = Path(__file__).parent.joinpath("_models")


merlinite7b.train("danceability", MODELS_SAVE_DIR, "tag_order", DATA_DIR)







# # Load the pre-trained model and tokenizer
# model_name = "ibm/merlinite-7b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = MistralForSequenceClassification.from_pretrained(model_name)

# classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# # Classify some text
# text = "This is a really good book, I think."
# result = classifier(text)
# print("Result", result)