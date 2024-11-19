import os
import logging
import pandas as pd
from helper.logging_config import setup_logging
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from helper.etl_func import compute_metrics  # Import compute_metrics

FEATURES_PATH = "./../data/features"
MODEL_PATH = "./../models/fine_tuned_bigbird_health_fact"
LOGS_PATH = "./../models/logs"

setup_logging("ingest.log")

def train_model(features_path: str, model_path: str, logs_path: str, num_labels: int = 4) -> None:
    """
    Train and evaluate a model on the preprocessed dataset.
    """
    try:
        # Load tokenized features
        logging.info(f"Loading features from {features_path}...")
        dataset = load_from_disk(features_path)
        tokenized_train = dataset["train"]
        tokenized_validation = dataset["validation"]
        tokenized_test = dataset["test"]

        # Load the model
        logging.info("Initializing BigBird model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "nbroad/bigbird-base-health-fact",
            num_labels=num_labels
        )

        # Training arguments
        logging.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=logs_path,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            logging_dir=os.path.join(logs_path, "logs"),
            logging_steps=50,
            push_to_hub=False
        )

        # Initialize Trainer
        logging.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,
            tokenizer=None,
            compute_metrics=compute_metrics  # Use imported compute_metrics
        )

        # Train the model
        logging.info("Starting training...")
        trainer.train()

        # Save the fine-tuned model
        logging.info(f"Saving fine-tuned model to {model_path}...")
        trainer.save_model(model_path)

        # Evaluate on the test set
        logging.info("Evaluating on the test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_test)
        
        # Save the test results
        save_test_results(test_results, logs_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)

    train_model(FEATURES_PATH, MODEL_PATH, LOGS_PATH)
