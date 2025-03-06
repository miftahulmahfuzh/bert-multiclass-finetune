import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import pickle
import hashlib
import wandb
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
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

def get_cache_path(split_name, config):
    """Generate cache path for a given split using hashing for uniqueness."""
    cache_dir = os.path.join(config["paths"]["data_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)

    config_str = f"{config['model']['name']}_{config['training']['max_length']}_{split_name}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()

    return os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pkl")

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

def evaluate_predictions(trainer, config, tokenizer):
    """
    Perform detailed prediction analysis and save results to Excel.
    """
    # Load test dataset
    test_dataset = load_dataset(config["data"]["dataset_name"])
    if 'test' not in test_dataset:
        print(f"NOT FOUND TEST SPLIT IN ORIGINAL DATASET")
        print(f"SPLITTING FROM TRAIN DATA..")
        test_dataset = test_dataset['train'].train_test_split(
            test_size=config["data"]["validation_split"]
        )['test']
    else:
        test_dataset = test_dataset['test']

    # Create test dataloader
    test_dataset = CachedInstructionDataset(
        test_dataset,
        tokenizer,
        config,
        cache_path=get_cache_path("test", config),
        split="test"
    )

    # Get predictions
    predictions = []
    texts = []
    targets = []
    model = trainer.model.eval()

    print("Generating predictions...")
    with torch.no_grad():
        for item in tqdm(test_dataset):
            # Get input text from the instruction prompt
            text = item['input_only']
            texts.append(text)

            # Get model prediction
            inputs = {
                'input_ids': item['input_ids'].unsqueeze(0).to(model.device),
                'attention_mask': item['attention_mask'].unsqueeze(0).to(model.device)
            }

            # outputs = model.generate(
            #     **inputs,
            #     max_new_tokens=config["training"]["max_new_tokens"],
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            #     num_return_sequences=1,
            #     temperature=0.1
            # )
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["training"]["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                temperature=0.7,           # Slightly higher temperature for creativity
                top_k=50,                  # Top-k sampling to limit to likely tokens
                top_p=0.9,                 # Top-p sampling for diversity
                repetition_penalty=1.2,    # Penalize repetition
                do_sample=True,            # Enable sampling for varied outputs
                no_repeat_ngram_size=2     # Prevent repeating 2-grams
            )

            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_text = pred_text.replace(item["text"], "")
            predictions.append(pred_text)

            # Get target text
            target_text = tokenizer.decode(item['labels'], skip_special_tokens=True)
            targets.append(target_text)

    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': targets,
        'predicted_label': predictions
    })

    # Save results
    output_path = os.path.join(config["paths"]["output_dir"], 'test_predictions.xlsx')
    results_df.to_excel(output_path, index=False)
    print(f"\nPrediction results saved to: {output_path}")

    # return results_df, accuracy
    return results_df

class CachedInstructionDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer,
            config,
            cache_path=None,
            split="",
        ):

        self.cache_path = cache_path
        self.tokenized_data = None
        self.prompt = open(config["paths"]["prompt"]).read()
        self.input_column = config["data"]["input_column"]
        self.output_column = config["data"]["output_column"]

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
            prompt = self.prompt.replace("<<INPUT>>", item[self.input_column])

            # Tokenize input
            input_encoding = tokenizer(
                prompt,
                padding='max_length',
                truncation="longest_first",
                max_length=max_length,
                return_tensors='pt'
            )

            # Tokenize target
            target_encoding = tokenizer(
                item[self.output_column],
                padding='max_length',
                truncation="longest_first",
                max_length=max_length,
                return_tensors='pt'
            )

            tokenized_item = {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten(),
                'text': prompt,
                'input_only': item[self.input_column]
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

    # If the mode is "test", skip training and directly evaluate
    if config["training"]["mode"] == "test":
        print("Running in test mode. Skipping training...")

        adapter_path = config["paths"]["trained_adapter"]
        output_path = "/".join(adapter_path.split("/")[:-1])
        test_config_path = f"{output_path}/finetune_config.json"
        config = load_config(test_config_path)
        print(f"Reloaded config from:\n{test_config_path}")

        config["paths"]["trained_adapter"] = adapter_path
        config["paths"]["output_dir"] = output_path

        # Initialize wandb
        a = adapter_path.split("/")[-2]
        wandb_name = f"{a}_test_only"
        wandb.init(
            project=config.get("wandb", {}).get("project", "instruction-tuning"),
            name=wandb_name,
            config=config
        )

        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the base model with 4-bit quantization for efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation='eager',
        )

        # print("Loading PEFT model...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded adapter from: {adapter_path}")

        # Evaluate on the test data
        print("\nGenerating detailed test predictions...")
        # results_df, test_accuracy = evaluate_predictions(model, config, tokenizer)
        results_df = evaluate_predictions(model, config, tokenizer)

        # Log final metrics to wandb
        wandb.log({
            # "final_test_accuracy": test_accuracy,
            "prediction_file": os.path.join(output_path, 'test_predictions.xlsx')
        })

        # Finish wandb run
        wandb.finish()

        print("Test evaluation completed successfully.")
        return

    # The rest of the training-related code will run if the mode is not "test"
    # Initialize wandb and other parts for training...

    # Initialize wandb
    wandb.init(
        project=config.get("wandb", {}).get("project", "instruction-tuning"),
        name=config["paths"]["output_dir"],
        config=config
    )
    config["wandb"]["url"] = wandb.run.url

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
        print(f"NOT FOUND VALIDATION SPLIT IN ORIGINAL DATASET")
        print(f"SPLITTING FROM TRAIN DATA..")
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

    # Custom trainer with logging
    class WandbTrainer(Trainer):
        def log(self, logs, *args, **kwargs):
            """
            Override the log method to send metrics to wandb
            """
            if self.state.global_step % self.state.logging_steps == 0:
                # Filter out non-numeric and irrelevant logs
                logs = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and k not in ['epoch', 'total_flos']}
                wandb.log(logs)
            super().log(logs, *args, **kwargs)

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
        report_to=["wandb"],  # Add wandb to report_to
    )

    # Initialize trainer
    trainer = WandbTrainer(
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


    # Train and save model
    print("Starting training...")
    trainer.train()
    trainer.save_model(os.path.join(config["paths"]["output_dir"], "best-checkpoint"))

    # Perform prediction analysis
    print("\nGenerating detailed test predictions...")
    # results_df, test_accuracy = evaluate_predictions(trainer, config, tokenizer)
    results_df = evaluate_predictions(trainer, config, tokenizer)

    # Log final metrics to wandb
    wandb.log({
        # "final_test_accuracy": test_accuracy,
        "prediction_file": os.path.join(config["paths"]["output_dir"], 'test_predictions.xlsx')
    })

    # Finish wandb run
    wandb.finish()

    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
