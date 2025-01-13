import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
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
    print(f"File descriptor limits - Soft: {new_soft}, Hard: {hard}")

def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

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
    config["wandb"]["run_name"] = outd
    config["paths"]["output_dir"] = outd
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

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])

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
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
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

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=dist.get_rank() != 0):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            # Explicit cleanup
            del outputs, predictions
            torch.cuda.empty_cache()

    # Gather predictions and labels from all processes
    all_preds = torch.tensor(all_preds, device=device)
    all_labels = torch.tensor(all_labels, device=device)

    gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)

    if dist.get_rank() == 0:
        # Concatenate all gathered predictions and labels
        all_preds = torch.cat(gathered_preds).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()

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
    else:
        accuracy = 0
        report_df = None

    # Average loss across all processes
    avg_loss = torch.tensor(total_loss / len(dataloader), device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()

    return avg_loss.item(), accuracy, report_df

def train(rank, world_size, CONFIG):
    setup_distributed(rank, world_size)

    # Record start time
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d_%H:%M:%S")

    if rank == 0:
        # Initialize wandb only on the main process
        wandb.init(
            project=CONFIG["wandb"]["project"],
            name=CONFIG["wandb"]["run_name"],
            config=CONFIG
        )
        CONFIG["wandb"]["url"] = wandb.run.url
        os.makedirs(CONFIG["paths"]["output_dir"], exist_ok=True)

        config_save_path = os.path.join(CONFIG["paths"]["output_dir"], "finetune_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(CONFIG, f, indent=3)
        print(f"Configuration saved to {config_save_path}")

    # Set device
    device = torch.device(f"cuda:{rank}")

    # Load data
    train_dataloader, val_dataloader, test_dataloader, tokenizer = load_and_prepare_data(rank, world_size, CONFIG)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model"]["name"],
        num_labels=CONFIG["model"]["num_labels"]
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

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

    best_accuracy = 0
    best_report_df = None
    for epoch in range(CONFIG["training"]["epochs"]):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{CONFIG['training']['epochs']}")

        # Set the epoch for the samplers
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        val_loss, accuracy, report_df = evaluate(model, val_dataloader, device)

        if rank == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy
            })

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print("\nClassification Report:")
            print(report_df)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_report_df = report_df
                print(f"New best accuracy: {best_accuracy:.4f} - Saving model...")

                model.module.save_pretrained(CONFIG["paths"]["output_dir"])
                tokenizer.save_pretrained(CONFIG["paths"]["output_dir"])

                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(),
                #     'best_accuracy': best_accuracy,
                # }, os.path.join(CONFIG["paths"]["output_dir"], 'training_state.pt'))

                # best_excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'best_checkpoint_classification_report.xlsx')
                # report_df.to_excel(best_excel_path, sheet_name="best_validation_report")

    # Final test evaluation
    if rank == 0:
        print("\nLoading the best model from the output directory for testing...")
        best_model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["paths"]["output_dir"],
            num_labels=CONFIG["model"]["num_labels"]
        ).to(device)
        # best_model = DDP(best_model, device_ids=[rank])

        test_loss, test_accuracy, test_report_df = evaluate(best_model, test_dataloader, device)
        print("\nTest Classification Report:")
        print(test_report_df)

        # Record end time and calculate duration
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
            'duration_second': [seconds]
        })

        # Save all reports to a single Excel file
        excel_path = os.path.join(config["paths"]["output_dir"], 'classification_reports.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            best_report_df.to_excel(writer, sheet_name='validation_report')
            test_report_df.to_excel(writer, sheet_name='test_report')
            duration_df.to_excel(writer, sheet_name='duration', index=False)

        print(f"\nAll reports saved to {excel_path}")
        print(f"Training duration - {hours}h {minutes}m {seconds}s")

        # test_excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'test_classification_report.xlsx')
        # test_report_df.to_excel(test_excel_path, sheet_name="test_report")

        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_classification_report": test_report_df,
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
    CONFIG = load_config()

    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Starting distributed training on {world_size} GPUs...")
        mp.spawn(train, args=(world_size, CONFIG), nprocs=world_size, join=True)
    else:
        print("Multi-GPU training requires at least 2 GPUs. Running on single GPU...")
        train(0, 1, CONFIG)

if __name__ == "__main__":
    main()
