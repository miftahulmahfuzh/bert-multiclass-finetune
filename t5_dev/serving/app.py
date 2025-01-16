import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    version="1.0.0"
)

# Global variables for the model and tokenizer
classifier = None
id2label = None
device = None

@app.on_event("startup")
async def load_model():
    global classifier, id2label, device

    # Retrieve model path from environment variable or use default
    # model_path = os.getenv("MODEL_PATH", "path_to_finetuned_t5_model")
    model_path = "/home/devmiftahul/nlp/t5_dev/google/mt5-base_20250116_114500"

    # Optionally, set device via environment variable
    use_gpu_env = os.getenv("USE_GPU", "True").lower()
    use_gpu = use_gpu_env in ["true", "1", "yes"]

    # Check if GPU is available
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        device_num = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device_num)}")
    else:
        device = torch.device("cpu")
        print("GPU not available or USE_GPU is set to False. Using CPU.")

    try:
        print(f"Loading T5 model from {model_path}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        print("Model and tokenizer loaded successfully!")

        # Initialize the pipeline for text2text-generation
        classifier = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.type == 'cuda' else -1
        )

        # Retrieve id2label mapping from model config
        if hasattr(model.config, 'id2label'):
            id2label = model.config.id2label
            print("Loaded id2label mapping from model config.")
        else:
            # If id2label is not in config, try to load from a separate file
            id2label_path = os.path.join(model_path, "id2label.json")
            if os.path.exists(id2label_path):
                import json
                with open(id2label_path, 'r') as f:
                    id2label = json.load(f)
                print("Loaded id2label mapping from id2label.json.")
            else:
                print("id2label mapping not found. Ensure that id2label is available.")
                raise FileNotFoundError("id2label mapping not found.")

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
    global classifier, id2label, device

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
            # Optionally, map label to a more readable format using id2label
            # This step assumes that the generated label corresponds to a key in id2label
            # If labels are directly readable, this mapping can be skipped
            # For example:
            # label = id2label.get(label, label)
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
