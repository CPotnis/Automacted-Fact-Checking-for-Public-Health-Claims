import os
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from helper.logging_config import setup_logging

# Set up logging
setup_logging("serve.log")

# Input schema
class ClaimRequest(BaseModel):
    claim: str

# Output schema
class ClaimResponse(BaseModel):
    veracity: str
    confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="Claim Veracity API",
    description="Application to predict the veracity of claims.",
    version="1.0.0",
)

MODEL_PATH = "./../models/fine_tuned_bigbird_health_fact"
MODEL_NAME = "nbroad/longformer-base-health-fact"

try:
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        logging.info(f"Loading model from {MODEL_PATH}.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        logging.info(f"Loading Hugging Face model: {MODEL_NAME}.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    model.eval()
    logging.info("Model and tokenizer successfully loaded.")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}", exc_info=True)
    raise RuntimeError(f"Error loading model or tokenizer: {e}")


# Endpoint
@app.post("/claim/v1/predict", response_model=ClaimResponse)
async def predict_claim(request: ClaimRequest):
    try:
        logging.info(f"Received claim: {request.claim}")

        # Tokenize input
        inputs = tokenizer(
            request.claim,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        logging.info("Input tokenized successfully.")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
            confidence, predicted_class = torch.max(probabilities, dim=0)
        logging.info(f"Inference completed. Predicted class: {predicted_class}, Confidence: {confidence.item()}")

        labels = ["False", "Mixture", "True", "Unproven"]  # Ensure these match your dataset's label order
        veracity = labels[predicted_class.item()]
        logging.info(f"Mapped prediction to label: {veracity}")

        return ClaimResponse(veracity=veracity, confidence=confidence.item())
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
