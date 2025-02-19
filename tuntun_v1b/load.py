import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load and preprocess data
fname = "./raw_data/News Category_v1b.xlsx"
sheet_name = "News List_Corporate Action"
df = pd.read_excel(fname, sheet_name=sheet_name)

# Select and clean columns
columns = ["Subcategory", "News Title", "News Content"]
df = df[columns]
df = df.dropna()

# Combine title and content
df['text'] = df['News Title'] + ' . ' + df['News Content']

# Create and save label mapping
unique_labels = sorted(df['Subcategory'].unique())
label_dict = {str(i): label for i, label in enumerate(unique_labels)}

with open('data/labels.json', 'w') as f:
    json.dump(label_dict, f, indent=4)

# Create reverse mapping for encoding
label_to_id = {v: k for k, v in label_dict.items()}

# Add label and label_str columns
df['label'] = df['Subcategory'].map(label_to_id)
df['label_str'] = df['Subcategory']

# Keep only necessary columns in correct order
df = df[['text', 'label', 'label_str']]

# Split the data
# First split into train and temp (80% train, 20% temp)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_str'])
# Split temp into validation and test (50% each, resulting in 10% of total each)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_str'])

# Save splits
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Create distribution DataFrame
def get_distribution(df, split_name):
    dist = df['label_str'].value_counts().reset_index()
    dist.columns = ['Subcategory', f'Count_{split_name}']
    dist[f'Percentage_{split_name}'] = (dist[f'Count_{split_name}'] / len(df) * 100).round(2)
    return dist

# Get distributions for all splits
all_dist = get_distribution(df, 'all')
train_dist = get_distribution(train_df, 'train')
val_dist = get_distribution(val_df, 'validation')
test_dist = get_distribution(test_df, 'test')

# Merge all distributions
final_dist = all_dist.merge(train_dist, on='Subcategory', how='outer')
final_dist = final_dist.merge(val_dist, on='Subcategory', how='outer')
final_dist = final_dist.merge(test_dist, on='Subcategory', how='outer')

# Sort by total count
final_dist = final_dist.sort_values('Count_all', ascending=False)

# Save distribution
final_dist.to_excel('data/label_distribution.xlsx', index=False)

# Print summary
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Number of unique categories: {len(unique_labels)}")
