from datetime import datetime
from tqdm import tqdm
import pandas as pd
import requests
import json
import os

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama server configuration
# BASE_URL = "http://localhost:11435"
BASE_URL = "http://localhost:11434"
# MODEL_NAME = "gemma-2b-classifier"
# MODEL_NAME = "llama3.3:70b"
# MODEL_NAME = "deepseek-r1:1.5b"
MODEL_NAME = "llama3.1:70b-instruct-q5_0"
DATASET_PATH = "/home/devmiftahul/nlp/llm_dev/comment_generation_19-05-2025/olama_inference/data/test_predictions.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_SHEET_NAME = "inference_result"

fprompt = "/home/devmiftahul/nlp/llm_dev/comment_generation_19-05-2025/prompts/prompt_v1.txt"
prompt = open(fprompt).read()

# Test texts
# test_texts = [
#     "Tech stocks are rising after quarterly earnings reports",
#     "Central bank announces new monetary policy measures",
#     "Renewable energy investments show significant growth"
# ]

def read_dataset(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Read the dataset from an Excel file."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if 'text' not in df.columns or 'true_label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'true_label' columns")
        return df
    except Exception as e:
        logger.error(f"Failed to read dataset: {str(e)}")
        raise

def test_ollama_inference():
    # Endpoint for model inference
    endpoint = f"{BASE_URL}/api/generate"

    # Prepare request payload
    payload = {
        "model": MODEL_NAME,
        # "prompt": f"Instruction: Categorize the news text\nInput: {test_texts[0]}\nResponse:",
        "prompt": prompt.replace("<<INPUT>>", test_texts[0]),
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

def bulk_inference(posts: list[str]):
    endpoint = f"{BASE_URL}/api/generate"
    results = []

    # for text in posts:
    for text in tqdm(posts, unit="post"):
        res = "failed to generate comment"
        if isinstance(text, str):
            if len(text.split()) >= 3:
                payload = {
                    "model": MODEL_NAME,
                    # "prompt": f"Instruction: Categorize the news text\nInput: {text}\nResponse:",
                    "prompt": prompt.replace("<<INPUT>>", text),
                    "stream": False
                }

                try:
                    response = requests.post(endpoint, json=payload)
                    response.raise_for_status()
                    res = response.json()['response']
                    x = f"post: {text}\ncomment: {res}"
                    logger.info(x)
                    # results.append(res)
                except requests.RequestException as e:
                    msg = f"Error: {e}\npost: {text}"
                    logger.error(msg)
                    res = msg
        results.append(res)

    # print("Bulk Inference Results:")
    # for text, result in zip(test_texts, results):
    #     print(f"Text: {text}\nComment: {result}\n")
    return results

def save_results(df: pd.DataFrame, comments: list, output_path: str):
    """Save the results to an Excel file."""
    try:
        # Create a copy of the input DataFrame
        result_df = df.copy()
        # Add generated comments
        result_df['generated_comment'] = comments

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Excel
        result_df.to_excel(output_path, sheet_name=OUTPUT_SHEET_NAME, index=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def main():
    # Read dataset
    logger.info(f"Reading dataset from {DATASET_PATH}")
    df = read_dataset(DATASET_PATH, SHEET_NAME)
    logger.info(f"Loaded {len(df)} rows from dataset")

    # Process each row individually
    logger.info(f"Starting processing for {len(df)} posts")
    comments = bulk_inference(df["text"].tolist())

    # Ensure comments align with input posts
    if len(comments) != len(df):
        logger.error(f"Mismatch: Generated {len(comments)} comments for {len(df)} posts")
        raise ValueError("Comment count does not match post count")

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"xlsx/{MODEL_NAME}_generated_comments_{timestamp}.xlsx"

    # Save results
    logger.info(f"Saving results to {output_path}")
    save_results(df, comments, output_path)
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    # Uncomment the method you want to test
    # test_ollama_inference()
    # bulk_inference()
    main()
