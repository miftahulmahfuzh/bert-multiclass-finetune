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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,  # Updated for T5
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp
import resource
import gc

# Increase system file descriptor limit
def increase_fd_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(hard, 65535)  # Increase soft limit up to hard limit or 65535
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    # print(f"File descriptor limits - Soft: {new_soft}, Hard: {hard}")

def setup_distributed(rank, world_size, CONFIG):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = CONFIG["nccl"]["host"]
    os.environ['MASTER_PORT'] = CONFIG["nccl"]["port"]
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def get_duration_df(start_time, best_epoch):
    # Record end time and calculate duration
    start_time_str = start_time.strftime("%Y-%m-%d_%H:%M:%S")
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d_%H:%M:%S")
    duration = end_time - start_time

    # Calculate duration components
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Create duration DataFrame
    duration_df = pd.DataFrame({
        'start': [start_time_str],
        'end': [end_time_str],
        'duration_hour': [hours],
        'duration_minute': [minutes],
        'duration_second': [seconds],
        'best_epoch': [best_epoch]
    })
    return duration_df, hours, minutes, seconds

def add_timestamp(caption):
    """Append a timestamp to the given caption."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{caption}_{timestamp}"

def load_config(config_path="finetune_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add timestamp to output directory
    outd = add_timestamp(config["model"]["name"])
    config["paths"]["output_dir"] = outd

    print(json.dumps(config, indent=3))

    # Load label mappings
    labels_path = f"{config['paths']['data_dir']}/labels.json"
    with open(labels_path, 'r') as f:
        id2label = json.load(f)
        # Create label2id mapping
        label2id = {v: int(k) for k, v in id2label.items()}

    # Add label mappings to config
    config["model"]["id2label"] = id2label
    config["model"]["label2id"] = label2id

    return config

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
    def __init__(self, texts, labels, tokenizer, max_length, cache_path=None, split="", CONFIG={}):
        self.cache_path = cache_path
        self.tokenized_data = None
        self.CONFIG = CONFIG

        if cache_path and os.path.exists(cache_path) and self.CONFIG["paths"]["use_cache"]:
            if dist.get_rank() == 0:
                print(f"Loading tokenized {split} data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        else:
            if dist.get_rank() == 0:
                print(f"Tokenizing {split} data...")
            self.tokenized_data = self._tokenize_and_cache(texts, labels, tokenizer, max_length)

    def _tokenize_and_cache(self, texts, labels, tokenizer, max_length):
        tokenized_data = []
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Tokenizing", disable=dist.get_rank() != 0):
            input_encoding = tokenizer(
                str(text),
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            # Prepare target text
            target_text = self.CONFIG["model"]["id2label"][str(label)]
            target_text = f"<category>{target_text}</category>"
            target_encoding = tokenizer(
                target_text,
                padding='max_length',
                truncation=True,
                max_length=self.CONFIG["training"]["label_max_length"],
                return_tensors='pt'
            )

            item = {
                'input_ids': input_encoding['input_ids'].flatten(),
                'attention_mask': input_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten(),
                'text': text  # Added to include original text
            }
            tokenized_data.append(item)

        if self.cache_path and self.CONFIG["paths"]["use_cache"] and dist.get_rank() == 0:
            print(f"Saving tokenized data to cache: {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(tokenized_data, f)

        return tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def load_and_prepare_data(rank, world_size, CONFIG):
    train_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "train.csv"))
    val_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "validation.csv"))
    test_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "test.csv"))

    if rank == 0:
        print(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"], from_tiktoken=False)

    # Create datasets with caching
    train_dataset = CachedNewsDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=CONFIG["training"]["max_length"],
        cache_path=get_cache_path("train", CONFIG),
        split="train",
        CONFIG=CONFIG,
    )

    val_dataset = CachedNewsDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=CONFIG["training"]["max_length"],
        cache_path=get_cache_path("validation", CONFIG),
        split="validation",
        CONFIG=CONFIG,
    )

    test_dataset = CachedNewsDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=CONFIG["training"]["max_length"],
        cache_path=get_cache_path("test", CONFIG),
        split="test",
        CONFIG=CONFIG,
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Prevent issues with uneven batch sizes
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        sampler=test_sampler,
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        drop_last=True
    )
    return train_dataloader, val_dataloader, test_dataloader, tokenizer

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", disable=dist.get_rank() != 0)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if dist.get_rank() == 0:
            progress_bar.set_postfix({"loss": loss.item()})

        # Explicit cleanup
        del outputs, loss
        torch.cuda.empty_cache()

    # Average loss across all processes
    avg_loss = torch.tensor(total_loss / len(dataloader), device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()

    return avg_loss.item()

# Updated evaluate function
def evaluate(
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        dataloader: DataLoader,
        device: str,
        CONFIG: dict,
        output_file="evaluation_results.tsv",
    ):
    model.eval()
    total_loss = 0
    all_results = []
    desc = "Testing" if "test" in output_file else "Evaluating"
    # tokenizer = model.module.get_tokenizer() if isinstance(model, DDP) else model.get_tokenizer()

    model_to_use = model
    if isinstance(model, DDP):
        model_to_use = model.module

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=dist.get_rank() != 0):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)
            texts = batch['text']  # Access original text

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Generate predictions
            generated_ids = model_to_use.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG["training"]["label_max_length"]
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            true_labels = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)

            for text, pred, label in zip(texts, preds, true_labels):
                all_results.append({
                    'text': text,
                    'prediction': pred,
                    'label': label
                })

            # Explicit cleanup
            del outputs, generated_ids
            torch.cuda.empty_cache()

    # Gather results from all processes
    if dist.is_initialized():
        # Convert list of dicts to list of tuples for gathering
        local_results = [f"{res['text']}\t{res['prediction']}\t{res['label']}" for res in all_results]
        gathered_results = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
        dist.gather_object(local_results, gathered_results, dst=0)

        if dist.get_rank() == 0:
            # Flatten the list of lists
            flat_results = []
            for proc_results in gathered_results:
                if proc_results is not None:
                    flat_results.extend(proc_results)
            # Convert back to list of dicts
            final_results = []
            for item in flat_results:
                text, pred, label = item.split('\t')
                final_results.append({
                    'text': text,
                    'prediction': pred,
                    'label': label
                })
            # Create DataFrame
            results_df = pd.DataFrame(final_results)
        else:
            results_df = None
    else:
        # Single-process evaluation
        results_df = pd.DataFrame(all_results)

    # Average loss across all processes
    avg_loss = torch.tensor(total_loss / len(dataloader), device=device)
    if dist.is_initialized():
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / dist.get_world_size()

    if dist.is_initialized() and dist.get_rank() == 0:
        # Save the DataFrame to CSV
        results_df.to_csv(output_file, index=False, sep="\t")
        return avg_loss.item(), results_df
    elif not dist.is_initialized():
        # Save the DataFrame to CSV
        results_df.to_csv(output_file, index=False, sep="\t")
        return avg_loss.item(), results_df
    else:
        return avg_loss.item(), None

def train(rank, world_size, CONFIG):
    setup_distributed(rank, world_size, CONFIG)

    # Record start time
    start_time = datetime.now()

    label2id = CONFIG["model"]["label2id"]
    id2label = CONFIG["model"]["id2label"]

    # Assign the device based on the provided GPU ID for this rank
    gpu_ids = CONFIG["training"]["gpu_ids"]
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)

    # Load data
    train_dataloader, val_dataloader, test_dataloader, tokenizer = load_and_prepare_data(rank, world_size, CONFIG)

    if rank == 0:
        # Initialize wandb only on the main process
        wandb.init(
            project=CONFIG["wandb"]["project"],
            name=CONFIG["paths"]["output_dir"],
            config=CONFIG
        )
        CONFIG["wandb"]["url"] = wandb.run.url
        os.makedirs(CONFIG["paths"]["output_dir"], exist_ok=True)

        label2id = CONFIG["model"].pop("label2id")
        id2label = CONFIG["model"].pop("id2label")

        config_save_path = os.path.join(CONFIG["paths"]["output_dir"], "finetune_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(CONFIG, f, indent=3)
        print(f"Configuration saved to {config_save_path}")

    # Initialize model
    model_config = AutoConfig.from_pretrained(
        CONFIG["model"]["name"],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        CONFIG["model"]["name"],
        config=model_config
    ).to(device)

    # Add a method to retrieve the tokenizer inside the model for evaluation
    # def get_tokenizer(self):
    #     return tokenizer  # Fixed to return the tokenizer passed to the dataset
    # model.get_tokenizer = get_tokenizer.__get__(model, type(model))

    # Wrap model with DDP
    model = DDP(model, device_ids=[gpu_ids[rank]])

    # Initialize optimizer and scheduler
    b1 = CONFIG["training"]["beta_1"]
    b2 = CONFIG["training"]["beta_2"]
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        betas=(b1, b2),
        eps=CONFIG["training"]["eps"],
        weight_decay=CONFIG["training"]["weight_decay"],
        amsgrad=CONFIG["training"]["amsgrad"]
    )
    total_steps = len(train_dataloader) * CONFIG["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["training"]["warmup_steps"],
        num_training_steps=total_steps
    )

    best_accuracy = 0  # This variable is no longer needed
    best_report_df = None  # This variable is no longer needed
    best_model = None  # This variable is no longer needed
    best_epoch = 0
    lowest_loss = 100
    for epoch in range(CONFIG["training"]["epochs"]):
        if rank == 0:
            print(f"\nStarting Epoch {epoch + 1}/{CONFIG['training']['epochs']}")

        # Set the epoch for the samplers
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        # Define output file paths for validation
        val_output_file = os.path.join(CONFIG["paths"]["output_dir"], f'validation_predictions_epoch_{epoch + 1}.tsv')
        val_loss, val_results_df = evaluate(
            model,
            tokenizer,
            val_dataloader,
            device,
            CONFIG,
            output_file=val_output_file
        )

        if rank == 0:
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                # "accuracy": accuracy  # Removed
            })
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation predictions saved to {val_output_file}")

            if val_loss < lowest_loss:
                print(f"New lowest val_loss: {val_loss} - Saving model...")
                lowest_loss = val_loss
                model.module.save_pretrained(CONFIG["paths"]["output_dir"])
                tokenizer.save_pretrained(CONFIG["paths"]["output_dir"])

    # After training, evaluate on test data
    if rank == 0:
        print("\nEvaluate model on test data")

    # Define output file path for test
    test_output_file = os.path.join(CONFIG["paths"]["output_dir"], 'test_predictions.tsv')
    test_loss, test_results_df = evaluate(
        model,
        tokenizer,
        test_dataloader,
        device,
        CONFIG,
        output_file=test_output_file
    )

    if rank == 0:
        print("\nTest evaluation completed.")
        print(f"Test predictions saved to {test_output_file}")

        # Record duration
        duration_df, hours, minutes, seconds = get_duration_df(start_time, best_epoch)
        excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'evaluation_reports.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Save validation and test DataFrames
            for epoch in range(CONFIG["training"]["epochs"]):
                e = epoch + 1
                val_file = os.path.join(CONFIG["paths"]["output_dir"], f'validation_predictions_epoch_{e}.tsv')
                if os.path.exists(val_file):
                    val_df = pd.read_csv(val_file, sep="\t")
                    print(f"val_df for epoch {e}:\n{val_df}")
                    sheet_name = f'validation_epoch_{e}'
                    val_df.to_excel(writer, sheet_name=sheet_name, index=False)
            if os.path.exists(test_output_file):
                test_df = pd.read_csv(test_output_file, sep="\t")
                test_df.to_excel(writer, sheet_name='test_predictions', index=False)
            # Save duration
            duration_df.to_excel(writer, sheet_name='duration', index=False)

        print(f"\nAll evaluation results saved to {excel_path}")
        print(f"Training duration - {hours}h {minutes}m {seconds}s")

        # Log final metrics to wandb
        wandb.log({
            "test_loss": test_loss,
            # "test_accuracy": test_accuracy,  # Removed
            # "test_classification_report": test_report_df,  # Removed
            "training_duration_hours": hours,
            "training_duration_minutes": minutes,
            "training_duration_seconds": seconds
        })

    cleanup_distributed()
    gc.collect()
    torch.cuda.empty_cache()

def main():
    increase_fd_limit()
    # Load configuration
    global CONFIG
    CONFIG = load_config()

    # Get the list of GPU IDs from the configuration
    gpu_ids = CONFIG["training"]["gpu_ids"]
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
        CONFIG["training"]["gpu_ids"] = gpu_ids

    # Validate GPU IDs
    available_gpus = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid >= available_gpus:
            raise ValueError(f"GPU id {gid} is not available. Available GPUs: {available_gpus}")

    world_size = len(gpu_ids)
    if world_size > 1:
        print(f"Starting distributed training on GPUs: {gpu_ids}")
        mp.spawn(train, args=(world_size, CONFIG), nprocs=world_size, join=True)
    else:
        print(f"Running on GPU: {gpu_ids[0]}")
        train(0, world_size, CONFIG)

if __name__ == "__main__":
    main()
