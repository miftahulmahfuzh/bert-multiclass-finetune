from datasets import load_dataset
import pandas as pd
import os

def load_and_save_dataset():
    # Create output directory if it doesn't exist
    output_dir = "financial_news_data"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset from Hugging Face
    dataset = load_dataset("intanm/indonesian-financial-topic-classification-dataset")

    # Convert each split to DataFrame and save to CSV
    for split in dataset.keys():
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset[split])

        # Save to CSV
        output_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {split} dataset to {output_path}")
        print(f"Number of samples in {split}: {len(df)}")

if __name__ == "__main__":
    load_and_save_dataset()
