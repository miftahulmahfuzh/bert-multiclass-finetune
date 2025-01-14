import os
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_config(config_path="finetune_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def add_timestamp(caption):
    """Append a timestamp to the given caption."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{caption}_{timestamp}"

class CachedNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }
        return item

def load_model_and_tokenizer(model_path, device):
    """Load the finetuned model and tokenizer from the specified path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer

def load_test_data(test_data_path, tokenizer, max_length, batch_size):
    """Load test data and create a DataLoader."""
    test_df = pd.read_csv(test_data_path)

    if 'text' not in test_df.columns or 'label' not in test_df.columns:
        raise ValueError("Test data must contain 'text' and 'label' columns.")

    texts = test_df['text'].values
    labels = test_df['label'].values

    test_dataset = CachedNewsDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_dataloader, test_df

def evaluate(model, dataloader, device):
    """Perform inference and evaluate the model on the test data."""
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    if 'accuracy' not in report_df.index:
        report_df.loc['accuracy'] = accuracy
    else:
        report_df.at['accuracy', 'precision'] = accuracy
        report_df.at['accuracy', 'recall'] = accuracy
        report_df.at['accuracy', 'f1-score'] = accuracy
        report_df.at['accuracy', 'support'] = len(all_labels)

    return avg_loss, accuracy, report_df

def save_evaluation_results(report_df, duration, output_file):
    """Save the evaluation report and duration to an Excel file."""
    # Create a duration DataFrame
    duration_df = pd.DataFrame({
        'start_time': [duration['start_time']],
        'end_time': [duration['end_time']],
        'duration': [duration['duration']],
        'best_epoch': [duration.get('best_epoch', 'N/A')]
    })

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='test_report')
        duration_df.to_excel(writer, sheet_name='duration', index=False)

    print(f"Evaluation results saved to {output_file}")

def main(finetuned_path: str, test_data_path: str):
    # Configuration
    config_path = f"{finetuned_path}/finetune_config.json"  # Replace with your config file path if different
    CONFIG = load_config(config_path)

    # Extract paths and parameters from config
    model_path = finetuned_path
    # test_data_path = CONFIG["paths"]["test_data_path"]
    # output_file = CONFIG["paths"].get("output_file", "test_result.xlsx")
    output_file = f"{finetuned_path}/test_result.xlsx"
    max_length = CONFIG["training"].get("max_length", 512)
    batch_size = CONFIG["training"].get("batch_size", 32)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # Load test data
    print("Loading test data...")
    test_dataloader, test_df = load_test_data(test_data_path, tokenizer, max_length, batch_size)

    # Perform evaluation
    print("Evaluating the model on test data...")
    start_time = datetime.now()
    avg_loss, accuracy, report_df = evaluate(model, test_dataloader, device)
    end_time = datetime.now()
    duration = end_time - start_time

    # Display evaluation metrics
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_df)

    # Prepare duration information
    duration_info = {
        'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
        'duration': str(duration),
        'best_epoch': CONFIG["training"].get("best_epoch", 'N/A')  # Placeholder if not applicable
    }

    # Save evaluation results
    save_evaluation_results(report_df, duration_info, output_file)

if __name__ == "__main__":
    finetuned_path = "/home/devmiftahul/nlp/bert_dev/bert-base-multilingual-uncased_20250114_120521"
    test_data_path = "/home/devmiftahul/nlp/bert_dev/financial_news_data/test.csv"
    main(finetuned_path, test_data_path)
