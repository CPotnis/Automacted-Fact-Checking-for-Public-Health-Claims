import os
import logging
from datasets import load_dataset
import logging
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def download_and_save_dataset(dataset: str, output_dir: str) -> None:
    """
    Downloads and saves a Hugging Face dataset.
    """
    try:
        logging.info(f"Downloading dataset '{dataset}'")
        data = load_dataset(dataset, trust_remote_code=True)

        os.makedirs(output_dir, exist_ok=True)
        for split, split_data in data.items():
            output_file = os.path.join(output_dir, f"{split}.csv")
            logging.info(f"Saving {split} split to {output_file}")
            split_data.to_csv(output_file)

        logging.info(f"Dataset saved to '{output_dir}'.")
    except Exception as e:
        logging.error("Error during dataset processing", exc_info=True)

def preprocess_function(examples, tokenizer, max_length):
    """
    Tokenize input examples.
    """
    inputs = [
        (claim if claim else "") + " " + (explanation if explanation else "")
        for claim, explanation in zip(examples["claim"], examples["explanation"])
    ]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)

def add_labels(examples):
    """
    Add labels to the dataset.
    """
    return {"labels": examples["label"]}


def compute_metrics(pred):
    """
    Compute evaluation metrics for the model.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    logging.info(f"Metrics computed: Accuracy={acc}, F1={f1}")
    return {"accuracy": acc, "f1": f1}

def save_test_results(test_results, logs_path):
    """
    Save test results to a CSV file.
    """
    try:
        results_df = pd.DataFrame([test_results])
        results_df_path = os.path.join(logs_path, "test_results.csv")
        results_df.to_csv(results_df_path, index=False)
        logging.info(f"Test results saved to {results_df_path}")
    except Exception as e:
        logging.error(f"Failed to save test results: {e}", exc_info=True)
