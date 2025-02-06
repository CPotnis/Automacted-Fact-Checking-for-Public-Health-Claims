# Automated Fact-Checking for Public Health Claims

This repository contains the implementation of an ML-based pipeline to automate the fact-checking of public health claims using the PUBHEALTH dataset. The project focuses on creating a robust and scalable solution that involves data collection, processing, model training, deployment, and monitoring.

## Overview

This project automates fact-checking for public health claims by utilizing pre-trained language models. The pipeline includes:
1. Ingesting the PUBHEALTH dataset.
2. Processing the data for fine-tuning.
3. Fine-tuning pre-trained models like `DistilBERT`, `Longformer`, and `BigBird`.
4. Deploying the trained model as an API using FastAPI.
5. Setting up monitoring to ensure model performance over time.

## Dataset

The PUBHEALTH dataset is used for this task. It contains:
- Claims with their associated veracity labels (`True`, `False`, `Unproven`, `Mixture`).
- Justifications for each label.

More details about the dataset can be found [here](https://huggingface.co/datasets/pubhealth).

## Pipeline Steps

### 1. Data Collection
- Script: `ingest.py`
- Downloads the PUBHEALTH dataset and saves it locally.

### 2. Data Processing
- Script: `prepare.py`
- Prepares the dataset for training by splitting it into training, validation, and test sets.

### 3. Model Selection
- Models used:
  - [DistilBERT](https://huggingface.co/austinmw/distilbert-base-uncased-finetuned-health_facts)
  - [Longformer](https://huggingface.co/nbroad/longformer-base-health-fact)
  - [BigBird](https://huggingface.co/nbroad/bigbird-base-health-fact)
- The final model is selected based on experimentation and compatibility with the dataset.

### 4. Training and Evaluation
- Script: `train.py`
- Fine-tunes the selected model with exposed hyperparameters.
- Includes evaluation scripts to measure model performance.

### 5. Deployment
- Script: `serve.py`
- Deploys the model using FastAPI with an endpoint `/claim/v1/predict`.
- A lightweight unit testing suite is included using `pytest`.
- Kubernetes deployment YAML is provided for container orchestration.

### 6. Monitoring and Updates
- Regular monitoring of the model using performance metrics.
- Updates are triggered when a significant drift in performance is detected.


## Contributing

Feel free to submit issues or pull requests to improve the pipeline or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
