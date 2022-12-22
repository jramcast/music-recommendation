from pathlib import Path

from transformers import RobertaForRegression, Trainer, EvaluationStrategy
from transformers.training import DataCollatorForRegression
from torch.utils.data import TensorDataset

import dataloading


DATA_DIR = Path(__file__).parent.joinpath("../../../../data/jaime_lastfm")

(
    X_train,
    y_train,
    X_validation,
    y_validation,
    _,
    _,
) = dataloading.read_tag_token_sets(
    DATA_DIR, 100, "danceability", "hours", "tag_weight"
)


model = BertForRegression(num_labels=1)


# Create training and validation datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_validation, y_validation)

# Define training and evaluation strategies
training_strategy = TrainingStrategy(loss_strategy="mean_absolute_error")
evaluation_strategy = EvaluationStrategy(metric_strategy="mean_absolute_error")

# Initialize trainer
trainer = Trainer(
    model=model,
    training_strategy=training_strategy,
    evaluation_strategy=evaluation_strategy,
    train_dataset=train_dataset,
    validation_dataset=val_dataset,
)

# Start training
trainer.train()

# Evaluate model on validation set
trainer.evaluate()