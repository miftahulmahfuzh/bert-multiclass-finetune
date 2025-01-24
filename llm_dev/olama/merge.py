import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Path to the base model and LoRA adapter
BASE_MODEL_PATH = "google/gemma-2-2b-it"
LORA_MODEL_PATH = "/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/best-checkpoint" # gemma-2b 20 epochs
MERGED_MODEL_PATH = f"{LORA_MODEL_PATH}/merged_model"
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)

# 1. Load base model
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_PATH,
#     torch_dtype=torch.float16,  # or float32
#     device_map="auto"
# )
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# 2. Load PEFT model
# peft_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# 3. Merge weights
# merged_model = peft_model.merge_and_unload()

# 4. Save merged model
# merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)
