import requests
import json

# Ollama server configuration
BASE_URL = "http://localhost:11435"
MODEL_NAME = "gemma-2b-classifier"

# Test texts
test_texts = [
    "Tech stocks are rising after quarterly earnings reports",
    "Central bank announces new monetary policy measures",
    "Renewable energy investments show significant growth"
]

def test_ollama_inference():
    # Endpoint for model inference
    endpoint = f"{BASE_URL}/api/generate"

    # Prepare request payload
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Instruction: Categorize the news text\nInput: {test_texts[0]}\nResponse:",
        "stream": False
    }

    # Send request
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()

        # Parse and print result
        result = response.json()
        print("Inference Result:", result['response'])

    except requests.RequestException as e:
        print(f"Error during inference: {e}")

def bulk_inference():
    endpoint = f"{BASE_URL}/api/generate"
    results = []

    for text in test_texts:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"Instruction: Categorize the news text\nInput: {text}\nResponse:",
            "stream": False
        }

        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            results.append(response.json()['response'])
        except requests.RequestException as e:
            results.append(f"Error: {e}")

    print("Bulk Inference Results:")
    for text, result in zip(test_texts, results):
        print(f"Text: {text}\nCategory: {result}\n")

if __name__ == "__main__":
    # Uncomment the method you want to test
    # test_ollama_inference()
    bulk_inference()
