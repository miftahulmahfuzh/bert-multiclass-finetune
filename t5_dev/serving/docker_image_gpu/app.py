import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import uvicorn
import torch
import re

# Define input schema
class PredictionInput(BaseModel):
    text: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="News Classification API",
    description="API for classifying news articles using a fine-tuned T5 model",
    version="0.1.0"
)

# Global variables for the model and tokenizer
classifier = None
device = None

@app.on_event("startup")
async def load_model():
    global classifier, device

    # Retrieve model path from environment variable or use default
    model_path = os.getenv("MODEL_PATH", "path_to_finetuned_t5_model")

    try:
        classifier = pipeline(
            "text2text-generation",
            model=model_path,
            device=0
        )

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

def extract_label(generated_text: str) -> str:
    """
    Extracts the label from the generated text.
    Assumes the label is wrapped within <category></category> tags.
    Example: "<category>Finance</category>" -> "Finance"
    """
    match = re.search(r"<category>(.*?)<\/category>", generated_text)
    if match:
        return match.group(1)
    else:
        # If tags are not present, return the entire generated text
        return generated_text.strip()

@app.post("/predict")
async def predict(input_data: PredictionInput):
    global classifier, device

    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not input_data.text:
        raise HTTPException(status_code=400, detail="No text provided for prediction")

    try:
        # Perform prediction using the pipeline
        results = classifier(input_data.text, max_length=32)  # Adjust max_length as needed

        predictions = []
        for result in results:
            generated_text = result['generated_text']
            label = extract_label(generated_text)
            predictions.append(label)

        return {
            "status": "success",
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    global classifier
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

if __name__ == "__main__":
    # Retrieve host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Optionally, set reload based on environment
    reload = os.getenv("RELOAD", "False").lower() in ["true", "1", "yes"]

    uvicorn.run("app:app", host=host, port=port, reload=reload)
