import requests
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Ollama server configuration
BASE_URL = "http://localhost:11435"
MODEL_NAME = "llama3.3:70b"

def load_prompt(prompt_file="prompt.txt"):
    """Load the prompt template from a file"""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {prompt_file} not found. Using default prompt.")
        return "summarize this news article to 2 sentences.\n<<TEXT>>"

def count_words(text):
    """Count the number of words in a text"""
    if not text:
        return 0
    return len(text.split())

def summarize_news(input_file, prompt_file="prompt.txt"):
    """
    Summarize news articles from a TSV file using Ollama API

    Parameters:
    -----------
    input_file : str
        Path to the input TSV file
    prompt_file : str, optional
        Path to the prompt template file
    """
    # Extract date from the filename
    news_date = os.path.basename(input_file).replace(".tsv", "")

    # Create output directories
    os.makedirs("summary_output", exist_ok=True)
    os.makedirs("summary_benchmark", exist_ok=True)

    # Load the TSV file
    input_df = pd.read_csv(input_file, sep='\t')[:3]

    # Load the prompt template
    prompt_template = load_prompt(prompt_file)

    # Prepare for benchmarking
    total_instances = len(input_df)
    start_time = time.time()

    input_word_counts = []
    output_word_counts = []
    durations = []
    results = []

    # Endpoint for model inference
    endpoint = f"{BASE_URL}/api/generate"

    # Process each article
    for idx, row in tqdm(input_df.iterrows(), total=total_instances, desc="Summarizing articles"):
        # Combine title and content
        title_content = f"Title: {row['title']}\n\nContent: {row['content']}"
        input_word_count = count_words(title_content)
        input_word_counts.append(input_word_count)

        # Prepare the prompt
        prompt = prompt_template.replace("<<TEXT>>", title_content)

        # Prepare request payload
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }

        # Send request and measure time
        api_start_time = time.time()
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            summary = response.json()['response']
            results.append(summary)

            # Count words in the output
            output_word_count = count_words(summary)
            output_word_counts.append(output_word_count)

        except requests.RequestException as e:
            print(f"Error for article {row['raw_id']}: {e}")
            summary = f"[ERROR] {str(e)}"
            results.append(summary)
            output_word_counts.append(0)

        api_end_time = time.time()
        duration = api_end_time - api_start_time
        durations.append(duration)

    # Record end time
    end_time = time.time()
    total_duration = end_time - start_time

    # Add summaries to the DataFrame
    input_df['summary'] = results

    # Save the results
    output_file = f"summary_output/{news_date}.tsv"
    input_df.to_csv(output_file, sep='\t', index=False)

    # Calculate benchmarks
    avg_duration = total_duration / total_instances if total_instances > 0 else 0
    avg_input_word_count = sum(input_word_counts) / total_instances if total_instances > 0 else 0
    avg_output_word_count = sum(output_word_counts) / total_instances if total_instances > 0 else 0

    # Min and max word counts
    min_input_word_count = min(input_word_counts) if input_word_counts else 0
    max_input_word_count = max(input_word_counts) if input_word_counts else 0
    min_output_word_count = min(output_word_counts) if output_word_counts else 0
    max_output_word_count = max(output_word_counts) if output_word_counts else 0

    # Format times for readability
    start_time_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

    benchmark_stats = {
        "news_date": news_date,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "total_instances": total_instances,
        "total_duration_seconds": round(total_duration, 2),
        "average_duration_seconds": round(avg_duration, 2),
        "avg_input_word_count": round(avg_input_word_count, 2),
        "min_input_word_count": min_input_word_count,
        "max_input_word_count": max_input_word_count,
        "avg_output_word_count": round(avg_output_word_count, 2),
        "min_output_word_count": min_output_word_count,
        "max_output_word_count": max_output_word_count,
        "min_duration_seconds": round(min(durations) if durations else 0, 2),
        "max_duration_seconds": round(max(durations) if durations else 0, 2)
    }

    # Write benchmark to file
    benchmark_file = f"summary_benchmark/{news_date}.txt"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        for key, value in benchmark_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"Summarization completed. Results saved to {output_file}")
    print(f"Benchmark saved to {benchmark_file}")

    return input_df

if __name__ == "__main__":
    # Automatically process all TSV files in the data directory
    data_dir = "data"

    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found.")
        exit(1)

    tsv_files = [f for f in os.listdir(data_dir) if f.endswith('.tsv')]

    if not tsv_files:
        print(f"No TSV files found in {data_dir} directory.")
        exit(1)

    for tsv_file in tsv_files:
        input_file = os.path.join(data_dir, tsv_file)
        print(f"Processing {input_file}...")
        summarize_news(input_file)
