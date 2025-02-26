# filename: download_llama3.py
from transformers import AutoModelForCausalLM, AutoTokenizer

token = ""

# Download the model and tokenizer
model_name = "atluzz/llama-3-8b-tuntun-en"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Save the model and tokenizer
model.save_pretrained("./atluzz_llama3_model")
tokenizer.save_pretrained("./atluzz_llama3_tokenizer")
