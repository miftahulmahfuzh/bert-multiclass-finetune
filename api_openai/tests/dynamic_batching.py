import pandas as pd
import requests
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000/mcp_chat"  # Adjust if your API is hosted elsewhere
DATASET_PATH = "data/test_predictions.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_SHEET_NAME = "generated_comments"
MAX_BATCH_ATTEMPTS = 5  # Maximum number of batch attempts to avoid infinite loops

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

def call_mcp_chat(posts: list) -> dict:
    """Call the /mcp_chat endpoint with a list of posts."""
    payload = {"inputs": posts}
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise

def process_batches(posts: list) -> list:
    """Process posts in batches, splitting as needed based on API response."""
    all_comments = []
    remaining_posts = posts.copy()
    batch_count = 0

    while remaining_posts and batch_count < MAX_BATCH_ATTEMPTS:
        batch_count += 1
        logger.info(f"Batch {batch_count}: Sending {len(remaining_posts)} posts")

        try:
            # Call API with current batch
            api_response = call_mcp_chat(remaining_posts)
            number_of_processed_posts = api_response.get('number_of_processed_posts', 0)
            comments = api_response.get('outputs', [])

            # Validate response
            if number_of_processed_posts != len(comments):
                logger.error(f"Batch {batch_count}: Mismatch between number_of_processed_posts ({number_of_processed_posts}) and comments length ({len(comments)})")
                raise ValueError("Invalid API response: comment count mismatch")

            logger.info(f"Batch {batch_count}: Processed {number_of_processed_posts} posts")

            # Add comments from this batch
            all_comments.extend(comments)

            # Update remaining posts
            if number_of_processed_posts < len(remaining_posts):
                logger.warning(f"Batch {batch_count}: Only {number_of_processed_posts} out of {len(remaining_posts)} posts processed")
                remaining_posts = remaining_posts[number_of_processed_posts:]
            else:
                # All posts processed
                remaining_posts = []

        except Exception as e:
            logger.error(f"Batch {batch_count}: Failed to process batch: {str(e)}")
            # Pad with empty comments for unprocessed posts in this batch
            all_comments.extend([''] * len(remaining_posts))
            break

    if remaining_posts:
        logger.error(f"Failed to process {len(remaining_posts)} posts after {batch_count} attempts")
        # Pad with empty comments for any remaining unprocessed posts
        all_comments.extend([''] * len(remaining_posts))

    return all_comments

def save_results(df: pd.DataFrame, comments: list, output_path: str):
    """Save the results to an Excel file."""
    try:
        # Create a copy of the input DataFrame
        result_df = df.copy()
        # Add generated comments
        result_df['generated_comment'] = comments
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
    posts = df['text'].tolist()
    logger.info(f"Loaded {len(posts)} posts from dataset")

    # Process posts in batches
    logger.info(f"Starting batch processing for {len(posts)} posts")
    comments = process_batches(posts)

    # Ensure comments align with input posts
    if len(comments) != len(posts):
        logger.error(f"Mismatch: Received {len(comments)} comments for {len(posts)} posts")
        raise ValueError("Comment count does not match post count")

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"api_result_{timestamp}.xlsx"

    # Save results
    logger.info(f"Saving results to {output_path}")
    save_results(df, comments, output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise
