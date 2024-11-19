import argparse
import logging
from helper.logging_config import setup_logging
from helper.etl_func import download_and_save_dataset

DEFAULT_OUTPUT_DIR = "./../data/raw"
DEFAULT_DATASET = "ImperialCollegeLondon/health_fact"

setup_logging("ingest.log")

def main():
    parser = argparse.ArgumentParser(
        description="Download and save a Hugging Face dataset."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=DEFAULT_DATASET,
        help="Name of the Hugging Face dataset to download."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the dataset will be saved."
    )
    args = parser.parse_args()

    download_and_save_dataset(args.dataset, args.output_dir)

if __name__ == "__main__":
    main()
