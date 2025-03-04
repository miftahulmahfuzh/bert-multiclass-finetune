import pandas as pd
from sklearn.model_selection import train_test_split

# Initial data loading (your code)
fname = "./raw_data/Collect Comment.xlsx"
df = pd.read_excel(fname, sheet_name="Comment-within post")
cols = ["Post Content", "Comment"]
df = df[cols]
df = df.dropna()
df.rename(columns={
    "Post Content": "post",
    "Comment": "comment"
    }, inplace=True)
print("Original dataframe:")
print(df)

# Set this flag to control duplicate post removal
remove_duplicate_post = False  # You can change this to False if needed
data_dir = "data"

# TODO 1: Remove duplicate posts if specified
if remove_duplicate_post:
    df = df.drop_duplicates(subset=['post'])
    data_dir = "data_no_dupl_post"
    print("\nAfter removing duplicate posts:")
    print(df)

# TODO 2: Split into train, dev, test (8:1:1) with no post overlap
# First, get unique posts
unique_posts = df['post'].unique()

# Split unique posts into train (80%) and remaining (20%)
train_posts, temp_posts = train_test_split(unique_posts,
                                         test_size=0.2,
                                         random_state=42)

# Split remaining into dev (10%) and test (10%)
dev_posts, test_posts = train_test_split(temp_posts,
                                       test_size=0.5,
                                       random_state=42)

# Create dataframes based on these post splits
train_df = df[df['post'].isin(train_posts)]
dev_df = df[df['post'].isin(dev_posts)]
test_df = df[df['post'].isin(test_posts)]

# Verify no overlap
print("\nVerification:")
print(f"Train posts in Dev: {len(set(train_df['post']) & set(dev_df['post']))}")
print(f"Train posts in Test: {len(set(train_df['post']) & set(test_df['post']))}")
print(f"Dev posts in Test: {len(set(dev_df['post']) & set(test_df['post']))}")
print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}")

# TODO 3: Save to CSV files
import os
os.makedirs(data_dir, exist_ok=True)

train_df.to_csv(f"{data_dir}/train.csv", index=False)
dev_df.to_csv(f"{data_dir}/dev.csv", index=False)
test_df.to_csv(f"{data_dir}/test.csv", index=False)

print(f"\nFiles saved in {data_dir}:")
print(f"train.csv: {len(train_df)} rows")
print(f"dev.csv: {len(dev_df)} rows")
print(f"test.csv: {len(test_df)} rows")
