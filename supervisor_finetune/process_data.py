import pandas as pd
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import re

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def clean_label(label):
    """Clean and normalize label text"""
    # Remove all non-alphanumeric characters (keeping spaces)
    label = label.split("\n")[0].lower()
    label = label.replace("agent", " ")
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', label)
    # Normalize whitespace and convert to lowercase
    return " ".join(cleaned.split())

def process_dataset(df):
    # Clean labels
    print("\nCleaning and normalizing labels...")
    original_labels = df['Agent'].unique()
    print("Original unique labels:", len(original_labels))
    for label in original_labels:
        cleaned_label = clean_label(label)
        print(f"Original: '{label}' -> Cleaned: '{cleaned_label}'")

    # Apply cleaning to the dataset
    df['Agent'] = df['Agent'].apply(clean_label)

    # Print initial label distribution and filter rare labels
    initial_label_dist = df['Agent'].value_counts()
    print("\nInitial label distribution after cleaning:")
    print(initial_label_dist)

    rare_labels = initial_label_dist[initial_label_dist < 3].index.tolist()
    if rare_labels:
        print(f"\nFiltering out labels with less than 3 samples:")
        for label in rare_labels:
            print(f"- {label}: {initial_label_dist[label]} samples")
        df = df[~df['Agent'].isin(rare_labels)]
        print(f"\nRemaining samples after filtering: {len(df)}")
    else:
        print("\nNo labels found with less than 3 samples")

    # 1. Get unique labels and create labels.json
    unique_labels = sorted(df['Agent'].unique())
    labels_dict = {str(i): label for i, label in enumerate(unique_labels)}

    with open('data/labels.json', 'w') as f:
        json.dump(labels_dict, f, indent=2)

    # 2. Create label2id mapping
    label2id = {label: str(i) for i, label in enumerate(unique_labels)}

    # 3. Create final dataset format
    processed_df = pd.DataFrame({
        'text': df['Questions'],
        'label': df['Agent'].map(label2id),
        'label_str': df['Agent']
    })

    # Split the data while preserving label distribution
    train_df, temp_df = train_test_split(processed_df, test_size=0.3, stratify=processed_df['label_str'], random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_str'], random_state=42)

    # Save splits to CSV
    train_df.to_csv('data/train.csv', index=False)
    dev_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    # 4. Calculate and save label distribution
    def get_distribution(data):
        return dict(Counter(data['label_str']))

    all_dist = get_distribution(processed_df)
    train_dist = get_distribution(train_df)
    dev_dist = get_distribution(dev_df)
    test_dist = get_distribution(test_df)

    # Create distribution DataFrame
    dist_df = pd.DataFrame({
        'all': pd.Series(all_dist),
        'train': pd.Series(train_dist),
        'dev': pd.Series(dev_dist),
        'test': pd.Series(test_dist)
    }).fillna(0).astype(int)

    # Add percentage columns
    for col in dist_df.columns:
        dist_df[f'{col}_pct'] = (dist_df[col] / dist_df[col].sum() * 100).round(2)

    # Save distribution to Excel
    dist_df.to_excel('data/label_distribution.xlsx')

    return {
        'total_samples': len(processed_df),
        'train_samples': len(train_df),
        'dev_samples': len(dev_df),
        'test_samples': len(test_df),
        'num_labels': len(unique_labels),
        'filtered_labels': rare_labels
    }

# Sample data (using your actual data structure)
# df = pd.DataFrame({
#     'Questions': [
#         'Sample question 1',
#         'Sample question 2',
#     ],
#     'Agent': [
#         'Stock Screener Agent',
#         'Stock-Screener'
#     ]
# })
fname = "/mnt/c/Users/mahfu/Downloads/tuntun/tuntun_ubuntu/hermawan/agentic_ai/dataset/agents.xlsx"
df = pd.read_excel(fname, sheet_name="Sheet1")

# Process the dataset
stats = process_dataset(df)

# Print processing summary
print("\nProcessing complete! Summary:")
print(f"Total samples: {stats['total_samples']}")
print(f"Train samples: {stats['train_samples']}")
print(f"Dev samples: {stats['dev_samples']}")
print(f"Test samples: {stats['test_samples']}")
print(f"Number of unique labels: {stats['num_labels']}")
if stats['filtered_labels']:
    print(f"Filtered labels: {', '.join(stats['filtered_labels'])}")
