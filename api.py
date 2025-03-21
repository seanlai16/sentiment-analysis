from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Optional
import uvicorn
from langdetect import detect
import numpy as np
import os

app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="API for sentiment analysis using fine-tuned BERT model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizer
MODEL_PATH = os.getenv("MODEL_PATH", "./saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict
    is_english: bool

def _detect_language(text: str) -> bool:
    """Helper function to detect language with error handling."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False
    
    try:
        return detect(text) == 'en'
    except Exception:
        return False

def predict_sentiment(text: str) -> dict:
    """Predict sentiment for given text."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check if text is English
    is_english = _detect_language(text)
    if not is_english:
        return {
            "sentiment": "unknown",
            "confidence": 0.0,
            "probabilities": {"positive": 0.0, "negative": 0.0},
            "is_english": False
        }
    
    # Tokenize input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors="pt"
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Convert to human-readable format
    sentiment = "positive" if predicted_class == 1 else "negative"
    probs = {
        "positive": float(probabilities[0][1]),
        "negative": float(probabilities[0][0])
    }
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probs,
        "is_english": True
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """Endpoint for sentiment prediction."""
    try:
        result = predict_sentiment(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_path": MODEL_PATH
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 