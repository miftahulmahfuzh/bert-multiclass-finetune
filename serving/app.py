from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import uvicorn
import os

# Define input schema
class PredictionInput(BaseModel):
    texts: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="News Classification API",
    description="API for classifying news articles using ONNX model",
    version="1.0.0"
)

API_KEY = "ac7c07ad4851146d36ba0af67ad8bfb5f945c694f122a0babb14ff2632b60196"

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Global variable for the pipeline
classifier = None

@app.on_event("startup")
async def load_model():
    global classifier
    # model_path = os.getenv("MODEL_PATH", "model_onnx")
    # model_path = "/home/devmiftahul/nlp/bert_dev/indobenchmark/indobert-base-p2_20250114_171614"
    # model_path = "/home/devmiftahul/nlp/bert_dev/Intel/dynamic_tinybert_20250131_093819"
    # model_path = "/home/devmiftahul/nlp/news_classification/tuntun_v1/indobenchmark/indobert-large-p2_20250217_155000"
    model_path = "/home/devmiftahul/nlp/news_classification/tuntun_v1/indobenchmark/indobert-lite-large-p2_20250218_104412"
    # model_path = "/home/devmiftahul/nlp/news_classification/tuntun_v1b/indobenchmark/indobert-large-p2_20250219_095826"
    # model_path = "Hello-SimpleAI/chatgpt-detector-roberta"
    # model = ORTModelForSequenceClassification.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
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

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(input_data: PredictionInput):
    global classifier

    try:
        # Check if model is loaded
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Make prediction
        predictions = classifier(input_data.texts)

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
