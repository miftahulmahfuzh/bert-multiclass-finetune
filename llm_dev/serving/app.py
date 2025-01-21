import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from typing import List

# Define request/response models
class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    categories: List[str]

# Initialize FastAPI app
app = FastAPI(title="Gemma Financial News Classifier")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    global model, tokenizer

    MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/gemma_2_2b_it/best-checkpoint"

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        print("Loading base model...")
        # Load the base model with 4-bit quantization for efficiency
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )

        print("Loading PEFT model...")
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_prediction(text: str) -> str:
    # Prepare the prompt
    prompt = f"Instruction: Categorize the news text\nInput: {text}\nResponse:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=1,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode prediction
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the category from the response
    try:
        category = predicted_text.split("Response:")[-1].strip()
    except:
        category = "Error processing response"

    return category

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Generate predictions for each text
        predictions = [generate_prediction(text) for text in request.texts]
        return PredictResponse(categories=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
