import logging
import os
from typing import Optional
from datasets import load_dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    pipeline,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint

from evaluation import calculate_metrics

from .interface import Model


class TransformerRegressor(Model):
    def __init__(
        self,
        model_name: str,
        target_column_name: str,
        text_column_name="tags",
        preprocessing_num_workers: Optional[int] = None,
        block_size: int = 256,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.target_column_name = target_column_name
        self.text_column_name = text_column_name
        self.preprocessing_num_workers = preprocessing_num_workers
        self.block_size = block_size
        self.logger = logger or logging.getLogger(__name__)
        self.training_args = self._parse_training_args()
        self.tokenizer = self._create_tokenizer()
        self.model = self._load_model()

        set_seed(self.training_args.seed)

    def _parse_training_args(self) -> TrainingArguments:
        parser = HfArgumentParser(TrainingArguments)
        (training_args,) = parser.parse_args_into_dataclasses()
        return training_args

    def fit(self, datasets: DatasetDict):
        self.logger.info("*** Tokenizing ***")
        tokenized_datasets = self._tokenize(datasets)

        train_dataset = tokenized_datasets["train"]
        validation_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            return calculate_metrics(labels, predictions)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )

        # Detecting last checkpoint.
        last_checkpoint = None
        if self._can_write_training_output():
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(self.training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir})"
                    " already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and self.training_args.resume_from_checkpoint is None
            ):
                self.logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. "
                    "To avoid this behavior, change "
                    "the `--output_dir` or add "
                    "`--overwrite_output_dir` to train from scratch."
                )

        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        self.logger.info("*** Training ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        self.logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(validation_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    def predict(self, text: str):
        pipe = pipeline("text-classification", model=self.training_args.output_dir)
        return pipe(text)

    def save(self):
        # TODO: use trainer save functs
        pass

    def load(self):
        pass

    def _can_write_training_output(self):
        return (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        )

    def _tokenize(self, datasets: DatasetDict):
        def tokenize_function(examples):
            label = examples[self.target_column_name]
            text = examples[self.text_column_name]

            result = self.tokenizer(
                text, padding=True, max_length=self.block_size, truncation=True
            )
            result["label"] = label

            return result

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=True,
        )

        return tokenized_datasets

    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            from_tf=False,
            config=self._generate_model_config(),
            revision="main",
        )

        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def _generate_model_config(self):
        config = AutoConfig.from_pretrained(
            self.model_name, num_labels=1, revision="main"
        )
        config.problem_type = "regression"
        config.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )

        return config

    def _create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision="main", use_fast=True
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer
