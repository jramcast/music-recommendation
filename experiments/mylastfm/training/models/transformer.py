from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification


import dataloading


DATA_DIR = Path(__file__).parent.joinpath("../../../../data/jaime_lastfm")

(X_train, y_train, X_validation, y_validation, _, _,) = dataloading.read_tag_token_sets(
    DATA_DIR, 100, "danceability", "hours", "tag_weight"
)


from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=1
)

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "csv",
    data_files=str(
        DATA_DIR.joinpath("merged_100_tokens_from_repeat_tags_str_by_hours.csv")
    ),
)

dataset = dataset.remove_columns("timestamp")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    print("examples")

    return tokenizer("holala", padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

exit()


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train()
