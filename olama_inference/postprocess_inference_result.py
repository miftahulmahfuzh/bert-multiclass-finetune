import pandas as pd
import os
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

d = "/mnt/c/Users/mahfu/Downloads/tuntun/tuntun_ubuntu/llm/comment_generation/olama_inference"
RAW_PATH = f"{d}/xlsx/gemma3:27b-it-qat_prompt_v2_20250523_181756.xlsx"
# RAW_PATH = f"{d}/xlsx/llama3.1-8b-vanilla_prompt_v2_20250527_134637.xlsx"
SHEET_NAME = "inference_result"
OUTPUT_SHEET_NAME = "inference_edited"

def read_dataset(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Read the dataset from an Excel file."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        cols = ["text", "true_label", "generated_comment"]
        if any(c not in df.columns for c in cols):
            raise ValueError("Dataset must contain 'text' and 'true_label' columns")
        return df
    except Exception as e:
        logger.error(f"Failed to read dataset: {str(e)}")
        raise

def get_first_xml_tag_content(comment_raw, tag="comment"):
    """
    Extract the content of the first XML tag occurrence using regex.

    Args:
        comment_raw (str): Raw comment text containing XML tags
        tag (str): XML tag name to search for (default: "comment")

    Returns:
        str: Content of the first XML tag, or original text if no tag found
    """
    if not isinstance(comment_raw, str):
        return str(comment_raw) if comment_raw is not None else ""

    # Pattern to match <tag>content</tag>
    pattern = f'<{tag}>(.*?)</{tag}>'

    # Find all matches
    matches = re.findall(pattern, comment_raw, re.DOTALL)

    if matches:
        # Return the content of the first match, stripped of leading/trailing whitespace
        return matches[0].strip()
    else:
        # If no XML tags found, return the original text
        return comment_raw

def generate_output_path(input_path: str) -> str:
    """
    Generate output path by modifying the input filename.

    Args:
        input_path (str): Original file path

    Returns:
        str: Modified output path with '_edited' suffix and in 'xlsx_edited' directory
    """
    path_obj = Path(input_path)

    # Get the filename without extension and add '_edited'
    filename_no_ext = path_obj.stem
    new_filename = f"{filename_no_ext}_edited{path_obj.suffix}"

    # Create the new directory path at the same level as xlsx folder
    # Go up one level from xlsx folder, then create xlsx_edited
    grandparent_dir = path_obj.parent.parent
    output_dir = grandparent_dir / "xlsx_edited"

    # Combine directory and filename
    output_path = output_dir / new_filename

    return str(output_path)

def save_results(df: pd.DataFrame, comments: list, output_path: str):
    """Save the results to an Excel file."""
    try:
        # Create a copy of the input DataFrame
        result_df = df.copy()
        # Add generated comments
        result_df['generated_comment'] = list(map(get_first_xml_tag_content, comments))

        # Keep only the 3 required columns
        result_df = result_df[['text', 'true_label', 'generated_comment']]

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
    logger.info(f"Reading dataset from {RAW_PATH}")
    df = read_dataset(RAW_PATH, SHEET_NAME)
    logger.info(f"Loaded {len(df)} rows from dataset")

    # Generate automatic output path
    output_path = generate_output_path(RAW_PATH)
    logger.info(f"Generated output path: {output_path}")

    # Extract comments from the 'generated_comment' column
    comments = df['generated_comment'].tolist()

    # Save results
    logger.info(f"Saving results to {output_path}")
    save_results(df, comments, output_path)
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
