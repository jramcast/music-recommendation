from pathlib import Path

import torch
import numpy as np
import dataloading
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Credit for the nice tutorial: https://towardsdatascience.com/linear-regression-with-hugging-face-3883fe729324

model_name = "bert-base-uncased"
max_length = 200
device = "cpu"
DATA_DIR = Path(__file__).parent.joinpath("../../../../data/jaime_lastfm")


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = (
        1
        / len(labels)
        * np.sum(2 * np.abs(logits - labels) / (np.abs(labels) + np.abs(logits)) * 100)
    )

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
) = dataloading.read_text_sets(DATA_DIR, "hours", "tag_weight", "danceability", True)

# Convert dataframe to list of strings as required by transformers
X_train = X_train.text.values.tolist()[:100]
y_train = y_train.values.tolist()[:100]
X_test = X_test.text.values.tolist()[:20]
y_test = y_test.values.tolist()[:20]


# Call the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the strings
train_encodings = tokenizer(
    X_train, truncation=True, padding=True, max_length=max_length
)
valid_encodings = tokenizer(
    X_test, truncation=True, padding=True, max_length=max_length
)


class MyLastFMTransformersDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # The "labels" property must contain float methods.
        # This is the target variable that we are predicting
        item["labels"] = float(torch.tensor([self.targets[idx]]))
        return item

    def __len__(self):
        return len(self.targets)


# convert our tokenized data into a torch Dataset
train_dataset = MyLastFMTransformersDataset(train_encodings, y_train)
valid_dataset = MyLastFMTransformersDataset(valid_encodings, y_test)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
model.to(device)


# Specifiy the arguments for the trainer
training_args = TrainingArguments(
    output_dir="./transformer_results",
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=20,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Call the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics_for_regression,
)

# Train the model
trainer.train()

# Call the summary
trainer.evaluate()
