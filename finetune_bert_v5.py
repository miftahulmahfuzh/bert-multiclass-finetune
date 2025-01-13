import os
import torch
import wandb
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

def load_config(config_path="finetune_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def add_timestamp(caption):
    """Append a timestamp to the given caption."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{caption}_{timestamp}"

# Load configuration
CONFIG = load_config()

# Add timestamp to run_name and output_dir
outd = add_timestamp(CONFIG["model"]["name"])
# CONFIG["wandb"]["run_name"] = outd
CONFIG["paths"]["output_dir"] = outd

# Initialize wandb
wandb.init(
    project=CONFIG["wandb"]["project"],
    name=CONFIG["wandb"]["run_name"],
    config=CONFIG
)

# Add wandb run URL to CONFIG
CONFIG["wandb"]["url"] = wandb.run.url

# Create output directory
os.makedirs(CONFIG["paths"]["output_dir"], exist_ok=True)

# Save the updated CONFIG to the output directory
config_save_path = os.path.join(CONFIG["paths"]["output_dir"], "finetune_config.json")
with open(config_save_path, 'w') as f:
    json.dump(CONFIG, f, indent=3)
print(f"Configuration saved to {config_save_path}")

def get_cache_path(split_name, config):
    """Generate cache path and create cache directory if it doesn't exist."""
    # Create cache directory
    cache_dir = os.path.join(config["paths"]["data_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash of the relevant configuration parameters
    config_str = f"{config['model']['name']}_{config['training']['max_length']}_{split_name}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()

    return os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pkl")

class CachedNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, cache_path=None, split=""):
        self.cache_path = cache_path
        self.tokenized_data = None

        # Try to load from cache first
        if cache_path and os.path.exists(cache_path) and CONFIG["paths"]["use_cache"]:
            print(f"Loading tokenized {split} data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        else:
            print(f"Tokenizing {split} data...")
            self.tokenized_data = self._tokenize_and_cache(texts, labels, tokenizer, max_length)

    def _tokenize_and_cache(self, texts, labels, tokenizer, max_length):
        tokenized_data = []
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Tokenizing"):
            encoding = tokenizer(
                str(text),
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(int(label))
            }
            tokenized_data.append(item)

        # Save to cache if cache_path is provided
        if self.cache_path and CONFIG["paths"]["use_cache"]:
            print(f"Saving tokenized data to cache: {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(tokenized_data, f)

        return tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def load_and_prepare_data():
    # Load CSV files
    train_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "train.csv"))
    val_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "validation.csv"))
    test_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "test.csv"))

    print(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])

    # Get cache paths
    train_cache_path = get_cache_path("train", CONFIG)
    val_cache_path = get_cache_path("validation", CONFIG)
    test_cache_path = get_cache_path("test", CONFIG)

    # Create datasets with caching
    train_dataset = CachedNewsDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=CONFIG["training"]["max_length"],
        cache_path=train_cache_path,
        split="train",
    )

    val_dataset = CachedNewsDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=CONFIG["training"]["max_length"],
        cache_path=val_cache_path,
        split="validation",
    )

    test_dataset = CachedNewsDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=CONFIG["training"]["max_length"],
        cache_path=test_cache_path,
        split="test",
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CONFIG["training"]["batch_size"]
    )

    return train_dataloader, val_dataloader, test_dataloader, tokenizer

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Generate classification report as a dictionary
    accuracy = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)

    # Convert the report dictionary to a pandas DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Add overall accuracy to the DataFrame if not already present
    if 'accuracy' not in report_df.index:
        report_df.loc['accuracy'] = accuracy
    else:
        # Update the accuracy value to ensure consistency
        report_df.at['accuracy', 'precision'] = accuracy
        report_df.at['accuracy', 'recall'] = accuracy
        report_df.at['accuracy', 'f1-score'] = accuracy
        report_df.at['accuracy', 'support'] = len(all_labels)

    return total_loss / len(dataloader), accuracy, report_df

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_dataloader, val_dataloader, test_dataloader, tokenizer = load_and_prepare_data()

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model"]["name"],
        num_labels=CONFIG["model"]["num_labels"]
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    total_steps = len(train_dataloader) * CONFIG["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["training"]["warmup_steps"],
        num_training_steps=total_steps
    )

    # Training loop
    best_accuracy = 0
    best_report = None

    for epoch in range(CONFIG["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['training']['epochs']}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)

        # Evaluate
        val_loss, accuracy, report_df = evaluate(model, val_dataloader, device)

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        })

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        # print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report_df)

        # Save best model
        if accuracy > best_accuracy:
            best_report_df = report_df
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.4f} - Saving model...")

            # Save model
            model.save_pretrained(CONFIG["paths"]["output_dir"])
            tokenizer.save_pretrained(CONFIG["paths"]["output_dir"])

            # Save additional information
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
            }, os.path.join(CONFIG["paths"]["output_dir"], 'training_state.pt'))

            # Save classification_report and accuracy for best_checkpoint in a dataframe to an excel file
            best_excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'best_checkpoint_classification_report.xlsx')
            best_report_df.to_excel(best_excel_path, sheet_name="best_validation_report")
            print(f"Best classification report and accuracy saved to {best_excel_path}")

        # Load model from CONFIG["paths"]["output_dir"]
        print("\nLoading the best model from the output directory for testing...")
        best_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["paths"]["output_dir"])
        best_model.to(device)

        # Call evaluate(model, test_dataloader, device)
        test_loss, test_accuracy, test_report_df = evaluate(best_model, test_dataloader, device)

        # Save classification_report and accuracy for best_checkpoint in a dataframe to an excel file
        test_excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'test_classification_report.xlsx')
        test_report_df.to_excel(test_excel_path, sheet_name="test_report")
        print(f"Test classification report and accuracy saved to {test_excel_path}")

        # Optionally, log test metrics to wandb
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_classification_report": test_report_df
        })

if __name__ == "__main__":
    main()
