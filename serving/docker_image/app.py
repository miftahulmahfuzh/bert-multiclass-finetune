from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from typing import List
import uvicorn
import os

# Define input schema
class PredictionInput(BaseModel):
    text: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="News Classification API",
    description="API for classifying news articles using ONNX model",
    version="1.0.0"
)

# Global variable for the pipeline
classifier = None

@app.on_event("startup")
async def load_model():
    global classifier
    model_path = os.getenv("MODEL_PATH", "model_onnx")
    model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        print(f"Loading ONNX model from {model_path}...")
        classifier = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    global classifier

    try:
        # Check if model is loaded
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Make prediction
        predictions = classifier(input_data.text)

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
    return {"status": "healthy", "model_loaded": classifier is not None}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
