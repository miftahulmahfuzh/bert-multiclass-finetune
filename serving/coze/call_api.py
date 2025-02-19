# Here, you can get input variables from the node through 'args' and output results through 'ret'
# 'args' and 'ret' have been correctly injected into the environment
# Example below: First get all input parameters from the node, then get the value of parameter named 'input':
# params = args.params;
# input = params.input;
# Example below: Output a 'ret' object containing multiple data types:
# ret: Output = { "name": 'John', "hobbies": ["reading", "traveling"] };

import requests
import json
from typing import List, Dict, Union

def test_classifier(texts: List[str]) -> Dict[str, Union[str, List[Dict[str, float]]]]:
    # API endpoint
    # url = "http://localhost:8000/predict"
    url = "http://10.183.0.2:8000/predict"
    # Prepare the request payload
    payload = {
        "text": texts
    }
    try:
        # Make the POST request
        response = requests.post(url, json=payload)
        # Raise an exception for bad status codes
        response.raise_for_status()
        # Parse and return the JSON response
        return response.json()
    except requests.RequestException as e:
        print(f"Error making request: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {e}")
        raise

async def main(args: Args) -> Output:
    params = args.params
    # Building output object example:
    # ret: Output = {
    #    "key0": params['input'] + params['input'], # Concatenate input parameter value twice
    #    "key1": ["hello", "world"],  # Output an array
    #    "key2": { # Output an Object
    #        "key21": "hi"
    #    },
    # }
    texts = [params['input']]
    r = test_classifier(texts)
    label = r["predictions"][0]['label']
    ret: Output = { "result": label }
    return ret

# Example usage
if __name__ == "__main__":
    sample_texts = [
        "F1 has always struggled to find a large US fan base, but finally found its secret: Americanize via @BW",
        "TOKYO NIKKEI STOCK AVERAGE OPENS 1.42% AT 26,892.73"
    ]
    try:
        result = test_classifier(sample_texts)
        print("Status:", result["status"])
        print("\nPredictions:")
        for pred in result["predictions"]:
            print(f"Label: {pred['label']}, Score: {pred['score']}")
    except Exception as e:
        print(f"An error occurred: {e}")
