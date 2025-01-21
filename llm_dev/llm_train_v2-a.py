import os
import torch
import json
import pickle
import hashlib
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset, DatasetDict
import argparse
import resource

# --- NEW IMPORTS FOR PEFT ---
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def increase_fd_limit():
    """Increase system file descriptor limit for handling large datasets."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(hard, 65535)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

def get_cache_path(split_name, model_name, max_length, data_dir="cache"):
    """Generate cache path for a given split using hashing for uniqueness."""
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    config_str = f"{model_name}_{max_length}_{split_name}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()

    return os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pkl")

class CachedInstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, cache_path=None, split=""):
        self.cache_path = cache_path
        self.tokenized_data = None
        self.prompt = "Categorize the news text"

        if cache_path and os.path.exists(cache_path):
            print(f"Loading tokenized {split} data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        else:
            print(f"Tokenizing {split} data...")
            self.tokenized_data = self._tokenize_and_cache(dataset, tokenizer, max_length)

    def _tokenize_and_cache(self, dataset, tokenizer, max_length):
        tokenized_data = []
        for item in tqdm(dataset, desc="Tokenizing"):
            # Create instruction prompt
            prompt = f"Instruction: {self.prompt}\nInput: {item['text']}\nResponse:"

            # Tokenize input
            input_encoding = tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            # Tokenize target
            target_encoding = tokenizer(
                item['label'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            tokenized_item = {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten(),
                'text': prompt  # keep original text for reference
            }
            tokenized_data.append(tokenized_item)

        if self.cache_path:
            print(f"Saving tokenized data to cache: {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(tokenized_data, f)

        return tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def prepare_dataloaders(dataset_dict, tokenizer, args):
    """Prepare DataLoaders with caching support."""
    train_dataset = CachedInstructionDataset(
        dataset_dict['train'],
        tokenizer,
        args.max_length,
        cache_path=get_cache_path("train", args.model_name, args.max_length, args.output_dir),
        split="train"
    )

    val_dataset = CachedInstructionDataset(
        dataset_dict['validation'],
        tokenizer,
        args.max_length,
        cache_path=get_cache_path("validation", args.model_name, args.max_length, args.output_dir),
        split="validation"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def main():
    parser = argparse.ArgumentParser(description="Enhanced Instruction Tuning Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset_name", type=str, default="Muennighoff/natural-instructions")
    parser.add_argument("--output_dir", type=str, default="./instruction-tuned-model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    args = parser.parse_args()

    # Increase file descriptor limit
    increase_fd_limit()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    print("Loading model with quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation='eager',
    )

    # --- PREPARE MODEL FOR k-bit TRAINING ---
    # This is necessary to ensure the model can be trained with LoRA on top of 4-bit weights.
    model = prepare_model_for_kbit_training(model)

    # --- SET UP LoRA ADAPTERS ---
    # Adjust LoRA hyperparameters and target_modules for your architecture and use-case.
    lora_config = LoraConfig(
        r=8,                         # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],  # typical for many GPT-like models; adjust for your arch
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("Loading dataset...")
    raw_datasets = load_dataset(args.dataset_name)

    # Create validation split if needed
    if "validation" not in raw_datasets:
        raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)
        dataset_dict = DatasetDict({
            "train": raw_datasets['train'],
            "validation": raw_datasets['test']
        })
    else:
        dataset_dict = raw_datasets

    # Prepare dataloaders with caching
    train_dataloader, val_dataloader = prepare_dataloaders(dataset_dict, tokenizer, args)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        save_total_limit=3,
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset,
        tokenizer=tokenizer,  # still pass tokenizer for preparing inputs
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=args.max_length,
            label_pad_token_id=-100,
        ),
    )

    # Train and save
    print("Starting training...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best-checkpoint"))
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
