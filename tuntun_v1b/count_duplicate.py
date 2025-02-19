import pandas as pd
import numpy as np

fname = "./raw_data/News Category_v1b.xlsx"
sheet_name = "News List_Corporate Action"
df = pd.read_excel(fname, sheet_name=sheet_name)

columns = ["Subcategory", "News Title", "News Content"]
df = df[columns]

# Find duplicates based on title
title_duplicates = df[df.duplicated(subset=['News Title'], keep=False)]

# Group by title and show all instances
print("\nDetailed Duplicate Title Analysis:")
print("-" * 80)

# Get unique titles and sort them, handling NaN separately
unique_titles = title_duplicates['News Title'].unique()
valid_titles = [t for t in unique_titles if isinstance(t, str)]
valid_titles.sort()

# First show non-NaN duplicates
for i, title in enumerate(valid_titles, start=1):
    matches = df[df['News Title'] == title]
    print(f"\n{i} - Title: {title}")
    print(f"Found in {len(matches)} rows.")
    print("Subcategories for each instance:")
    for idx, row in matches.iterrows():
        print(f"  Index {idx+2}: {row['Subcategory']}")
    print("-" * 80)

# Summary statistics
print("\nSummary:")
print(f"Total number of duplicate titles (excluding NaN): {len(valid_titles)}")

# Analyze NaN by subcategory
print("\nNaN Analysis by Subcategory:")
print("-" * 80)

for subcategory in sorted(df['Subcategory'].unique()):
    subset = df[df['Subcategory'] == subcategory]
    nan_titles = subset['News Title'].isna().sum()
    nan_content = subset['News Content'].isna().sum()

    if nan_titles > 0 or nan_content > 0:
        print(f"\nSubcategory: {subcategory}")
        if nan_titles > 0:
            print(f"  NaN Titles: {nan_titles}")
        if nan_content > 0:
            print(f"  NaN Content: {nan_content}")

print("\nTotal NaN counts:")
print(df.isna().sum())
