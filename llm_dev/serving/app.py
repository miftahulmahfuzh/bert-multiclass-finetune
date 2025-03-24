import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel, PeftConfig
from utils import extract_xml_content

# Define request/response models
class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    categories: List[str]

# Initialize FastAPI app
app = FastAPI(title="LLM Text Generation API")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt_path = "/home/devmiftahul/nlp/llm_dev/comment_generation/prompts/prompt_v1.txt"

@app.on_event("startup")
async def load_model():
    global model, tokenizer

    # MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/gemma_2_2b_it/best-checkpoint"
    # MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250121_144828/best-checkpoint" # gemma-2b 1 epoch
    # MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/best-checkpoint" # gemma-2b 20 epochs
    # base_model = "google/gemma-2-2b-it"

    # MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/v3/meta-llama/Llama-3.2-1B-Instruct_20250124_173732/best-checkpoint" # llama-3.2-1b 20 epochs
    # base_model = "meta-llama/Llama-3.2-1B-Instruct"

    # MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/v3/meta-llama/Llama-3.2-3B-Instruct_20250124_172327/checkpoint-27450"
    # base_model = "meta-llama/Llama-3.2-3B-Instruct"

    # MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/v3/mistralai/Mistral-Nemo-Instruct-2407_20250127_133602/checkpoint-23000"
    # base_model = "mistralai/Mistral-Nemo-Instruct-2407"

    MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/comment_generation/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_20250304_135543/best-checkpoint"
    # base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    base_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    try:
        print("Loading tokenizer...")
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        print("Loading base model...")
        # Load the base model with 4-bit quantization for efficiency
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     base_model,
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        #     load_in_4bit=True,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=quantization_config,
            attn_implementation='eager',
        )


        # print("Loading PEFT model...")
        # model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_prediction(text: str) -> str:
    # Prepare the prompt
    # prompt = f"Instruction: Categorize the news text\nInput: {text}\nResponse:"
    global prompt_path
    prompt = open(prompt_path).read()
    prompt = prompt.replace("<<INPUT>>", text)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_beams=1,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode prediction
    # predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # default without postprocessing

    # with postprocessing
    predicted_text = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:])
    predicted_text = extract_xml_content(predicted_text, tag="comment")

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
