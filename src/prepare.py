import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from helper.logging_config import setup_logging
import logging
from helper.etl_func import preprocess_function, add_labels  # Import functions from retl_func.py

INPUT_DIR = "./../data/raw"
OUTPUT_DIR = "./../data/features"
MODEL_NAME = "nbroad/bigbird-base-health-fact"
MAX_LENGTH = 512

# Setup logging
setup_logging("preprocess_dataset.log")

def preprocess_dataset(input_dir: str, output_dir: str, model_name: str, max_length: int) -> None:
    """
    Preprocesses the dataset.
    """
    try:
        logging.info(f"Loading raw dataset from {input_dir}.")
        dataset = DatasetDict({
            "train": load_dataset("csv", data_files=os.path.join(input_dir, "train.csv"))["train"],
            "validation": load_dataset("csv", data_files=os.path.join(input_dir, "validation.csv"))["train"],
            "test": load_dataset("csv", data_files=os.path.join(input_dir, "test.csv"))["train"]
        })

        logging.info(f"Initializing tokenizer for {model_name}.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logging.info("Tokenizing dataset.")
        tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True)

        logging.info("Adding labels to tokenized dataset.")
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

        logging.info("Formatting dataset for PyTorch.")
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Saving processed features to {output_dir}.")
        tokenized_dataset.save_to_disk(output_dir)
        logging.info(f"Features saved successfully in {output_dir}.")
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}", exc_info=True)


if __name__ == "__main__":
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR, MODEL_NAME, MAX_LENGTH)
