import json

fname = "clean-newsletter.json"
d = json.load(open(fname))
print(len(d))
print(d.keys())
print(d["published_date"]["68392"])
print(d["title"]["68392"])
print(d["content"]["68392"])

import json
import pandas as pd
import os
from datetime import datetime

def fetch_by_date(date_str, fname="clean-newsletter.json"):
    """
    Fetch articles published on a specific date from the JSON data file
    and save them to a TSV file.

    Parameters:
    -----------
    date_str : str
        Date string in format 'YYYY-MM-DD'
    fname : str, optional
        Path to the JSON data file (default: "clean-newsletter.json")

    Returns:
    --------
    pd.DataFrame
        DataFrame containing raw_id, title, content, and link for matching articles
    """
    # Load the JSON data
    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create lists to store the matching articles
    raw_ids = []
    titles = []
    contents = []
    links = []

    # Iterate through the published_date keys to find matching dates
    for raw_id, published_date in data["published_date"].items():
        # Extract just the date part (YYYY-MM-DD) from the datetime string
        article_date = published_date.split()[0]

        if article_date == date_str:
            # Add matching articles to our lists
            raw_ids.append(raw_id)
            titles.append(data["title"].get(raw_id, ""))
            contents.append(data["content"].get(raw_id, ""))
            links.append(data["link"].get(raw_id, ""))

    # Create a DataFrame with the required columns
    df = pd.DataFrame({
        "raw_id": raw_ids,
        "title": titles,
        "content": contents,
        "link": links
    })

    # Create the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save the DataFrame to a TSV file
    output_file = f"data/{date_str}.tsv"
    df.to_csv(output_file, sep='\t', index=False)

    print(f"Saved {len(df)} articles to {output_file}")

    return df

# Example usage
df = fetch_by_date("2025-01-13")
