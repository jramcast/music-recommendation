import logging
import os
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    pipeline
)
from transformers.testing_utils import CaptureLogger

MODEL_NAME = "gpt2"
TRAIN_FILE = "dataset_train.csv"
VALIDATION_FILE = "dataset_validation.csv"
PREPROCESSING_NUM_WORKERS = None
BLOCKSIZE = 4


logger = logging.getLogger(__name__)

parser = HfArgumentParser(TrainingArguments)
(training_args,) = parser.parse_args_into_dataclasses()

# Set seed before initializing model.
set_seed(training_args.seed)

config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1, revision="main")
config.problem_type = "regression"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision="main", use_fast=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, from_tf=False, config=config, revision="main"
)


model.resize_token_embeddings(len(tokenizer))

# Build datasets
datasets = load_dataset(
    "csv", data_files={"train": TRAIN_FILE, "validation": VALIDATION_FILE}
)


# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger(
    "transformers.tokenization_utils_base"
)


def tokenize_function(examples):
    label = examples["danceability"]
    text = examples["tags"]

    result = tokenizer(text, padding=True, max_length=tokenizer.model_max_length, truncation=True)
    result["label"] = label

    return result


lm_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=PREPROCESSING_NUM_WORKERS,
    remove_columns=column_names,
    load_from_cache_file=True,
)

if BLOCKSIZE > tokenizer.model_max_length:
    logger.warning(
        f"The block_size passed ({BLOCKSIZE}) is larger "
        "than the maximum length for the model"
        f"({tokenizer.model_max_length}). "
        "Using block_size={tokenizer.model_max_length}."
    )
block_size = min(BLOCKSIZE, tokenizer.model_max_length)


# # Pad line by line
# def prepare_sentence_and_target(examples):
#     result = tokenizer.pad(examples, padding=True)
#     result["labels"] = result["input_ids"].copy()
#     return result


# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

# lm_datasets = tokenized_datasets.map(
#     prepare_sentence_and_target,
#     batched=True,
#     num_proc=PREPROCESSING_NUM_WORKERS,
#     load_from_cache_file=True,
# )

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
)


# Detecting last checkpoint.
last_checkpoint = None
if (
    os.path.isdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


logger.info("*** Evaluate ***")

metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


logger.info("*** Example ***")

pipe = pipeline("text-classification", model=training_args.output_dir)

tags = "Hello I am Jaime"
print(tags, pipe(tags))
