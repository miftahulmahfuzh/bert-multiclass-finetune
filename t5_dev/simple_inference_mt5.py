import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

def classify_news(prompt, text_input):
    """
    Classify news text using MT5 model

    Args:
        prompt (str): Classification prompt
        text_input (str): News text to classify

    Returns:
        str: Predicted category
    """
    # Load model and tokenizer
    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    t = model.get_tokenizer()
    print(f"T: {t}")

    # Prepare input text
    input_text = f"{prompt} {text_input}"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,
            num_beams=4,
            early_stopping=True
        )

    # Decode prediction
    predicted_category = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_category

# Example usage
prompt = "classify text_input into the most appropriate category"
text_input = "RATA-RATA SAHAM NIKKEI TOKYO DIBUKA 1,42% PADA 26.892,73"

predicted_category = classify_news(prompt, text_input)
print(f"Input text: {text_input}")
print(f"Predicted category: {predicted_category}")
