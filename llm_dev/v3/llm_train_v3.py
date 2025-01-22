import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
import pickle
import hashlib
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
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
import resource
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def increase_fd_limit():
    """Increase system file descriptor limit for handling large datasets."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(hard, 65535)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

def add_timestamp(name):
    """Append a timestamp to the given name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}"

def load_config(config_path="finetune_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add timestamp to output directory
    outd = add_timestamp(config["model"]["name"])
    config["paths"]["output_dir"] = outd

    print(json.dumps(config, indent=3))
    return config

def get_cache_path(split_name, config):
    """Generate cache path for a given split using hashing for uniqueness."""
    cache_dir = os.path.join(config["paths"]["data_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)

    config_str = f"{config['model']['name']}_{config['training']['max_length']}_{split_name}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()

    return os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pkl")

class CachedInstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, config, cache_path=None, split=""):
        self.cache_path = cache_path
        self.tokenized_data = None
        self.prompt = config["training"]["instruction_prompt"]

        if cache_path and os.path.exists(cache_path) and config["paths"]["use_cache"]:
            print(f"Loading tokenized {split} data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        else:
            print(f"Tokenizing {split} data...")
            self.tokenized_data = self._tokenize_and_cache(dataset, tokenizer, config["training"]["max_length"])

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
                'text': prompt
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

def prepare_dataloaders(dataset_dict, tokenizer, config):
    """Prepare DataLoaders with caching support."""
    train_dataset = CachedInstructionDataset(
        dataset_dict['train'],
        tokenizer,
        config,
        cache_path=get_cache_path("train", config),
        split="train"
    )

    val_dataset = CachedInstructionDataset(
        dataset_dict['validation'],
        tokenizer,
        config,
        cache_path=get_cache_path("validation", config),
        split="validation"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def main():
    # Load configuration
    config = load_config()

    # Increase file descriptor limit
    increase_fd_limit()

    # Create output directory
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)

    # Save the configuration
    config_save_path = os.path.join(config["paths"]["output_dir"], "finetune_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=3)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
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
        config["model"]["name"],
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation='eager',
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Set up LoRA adapters
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("Loading dataset...")
    raw_datasets = load_dataset(config["data"]["dataset_name"])

    # Create validation split if needed
    if "validation" not in raw_datasets:
        raw_datasets = raw_datasets['train'].train_test_split(
            test_size=config["data"]["validation_split"]
        )
        dataset_dict = DatasetDict({
            "train": raw_datasets['train'],
            "validation": raw_datasets['test']
        })
    else:
        dataset_dict = raw_datasets

    # Prepare dataloaders with caching
    train_dataloader, val_dataloader = prepare_dataloaders(dataset_dict, tokenizer, config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["paths"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        evaluation_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        save_total_limit=config["training"]["save_total_limit"],
        report_to=config["training"]["report_to"],
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=config["training"]["max_length"],
            label_pad_token_id=-100,
        ),
    )

    # Train and save
    print("Starting training...")
    trainer.train()
    trainer.save_model(os.path.join(config["paths"]["output_dir"], "best-checkpoint"))
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
