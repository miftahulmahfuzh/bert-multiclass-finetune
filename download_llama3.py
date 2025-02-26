# filename: download_llama3.py
from transformers import AutoModelForCausalLM, AutoTokenizer

token = ""

# Download the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Save the model and tokenizer
model.save_pretrained("./llama3_model")
tokenizer.save_pretrained("./llama3_tokenizer")
