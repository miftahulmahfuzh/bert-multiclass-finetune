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
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
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

    T = AutoTokenizer
    if "indobenchmark" in CONFIG["model"]["name"]:
        T = BertTokenizer
    tokenizer = T.from_pretrained(CONFIG["model"]["name"])

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
        # drop_last=True  # Prevent issues with uneven batch sizes
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        sampler=test_sampler,
        num_workers=2,  # Reduced number of workers
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # drop_last=True
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

def evaluate(
        model,
        dataloader,
        device,
        labels=None,
        target_names=None,
        is_test=False,
        test_df=None,
        id2label={}
    ):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []  # Store the original texts
    desc = "Testing" if is_test else "Evaluating"
    # if not test_df:
    #     print(test_df)
    #     raise ValueError(f"EMPTY TEST_DF")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, disable=dist.get_rank() != 0)):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            # Store the original texts from the test DataFrame
            if test_df is not None:
                # Calculate the actual indices in the original DataFrame
                start_idx = batch_idx * dataloader.batch_size
                end_idx = start_idx + len(predictions)
                all_texts.extend(test_df['text'].iloc[start_idx:end_idx].values)

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
        report_dict = classification_report(
            all_labels,
            all_preds,
            labels=labels, # Specify the label IDs
            target_names=target_names, # Specify the label names
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()

        if 'accuracy' not in report_df.index:
            report_df.loc['accuracy'] = accuracy
        else:
            report_df.at['accuracy', 'precision'] = accuracy
            report_df.at['accuracy', 'recall'] = accuracy
            report_df.at['accuracy', 'f1-score'] = accuracy
            report_df.at['accuracy', 'support'] = len(all_labels)

        # Modify the predictions dictionary creation
        prd = {
            'text': all_texts,
            'true_label_id': all_labels,
            'predicted_label_id': all_preds,
            'true_label': [id2label[str(label)] for label in all_labels],
            'predicted_label': [id2label[str(pred)] for pred in all_preds],
            'correct': np.array(all_labels) == np.array(all_preds)
        }
        # print(f"LEN ALL_TEXTS {len(all_texts)}")
        # print(f"LEN ALL_LABELS {len(all_labels)}")
        # print(f"LEN ALL_PREDS {len(all_preds)}")
        # print(f"LEN ALL_TRUE_LABEL {len(prd['true_label'])}")
        # print(f"LEN ALL_PRED_LABEL {len(prd['predicted_label'])}")
        # print(f"LEN ALL_CORRECT {len(prd['correct'])}")
        predictions_df = pd.DataFrame(prd)
    else:
        accuracy = 0
        report_df = None
        predictions_df = None

    # Average loss across all processes
    avg_loss = torch.tensor(total_loss / len(dataloader), device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()

    return avg_loss.item(), accuracy, report_df, predictions_df

def train(rank, world_size, CONFIG):
    setup_distributed(rank, world_size, CONFIG)

    # Record start time
    start_time = datetime.now()

    label2id = CONFIG["model"].pop("label2id")
    id2label = CONFIG["model"].pop("id2label")

    # Prepare label information for classification_report
    num_labels = len(id2label)
    labels_sorted = list(range(num_labels))  # Assuming labels are 0 to num_labels-1
    target_names = [id2label[str(i)] for i in labels_sorted]  # Convert IDs to label names

    if rank == 0:
        # Initialize wandb only on the main process
        wandb.init(
            project=CONFIG["wandb"]["project"],
            name=CONFIG["paths"]["output_dir"],
            config=CONFIG
        )
        CONFIG["wandb"]["url"] = wandb.run.url
        os.makedirs(CONFIG["paths"]["output_dir"], exist_ok=True)


        config_save_path = os.path.join(CONFIG["paths"]["output_dir"], "finetune_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(CONFIG, f, indent=3)
        print(f"Configuration saved to {config_save_path}")

    # Assign the device based on the provided GPU ID for this rank
    gpu_ids = CONFIG["training"]["gpu_ids"]
    device = torch.device(f"cuda:{gpu_ids[rank]}")

    # Load data
    train_dataloader, val_dataloader, test_dataloader, tokenizer = load_and_prepare_data(rank, world_size, CONFIG)

    # Initialize model
    model_config = AutoConfig.from_pretrained(
        CONFIG["model"]["name"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model"]["name"],
        config=model_config
    ).to(device)

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

    best_accuracy = 0
    best_report_df = None
    best_model = None
    best_epoch = 0
    for epoch in range(CONFIG["training"]["epochs"]):
        if rank == 0:
            print(f"\nStarting Epoch {epoch + 1}/{CONFIG['training']['epochs']}")

        # Set the epoch for the samplers
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        val_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "validation.csv"))
        val_loss, accuracy, report_df, val_predictions_df = evaluate(
            model,
            val_dataloader,
            device,
            labels=labels_sorted, # Pass label IDs
            target_names=target_names, # Pass label names
            test_df=val_df,
            id2label=id2label
        )

        best_model = model
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
                best_epoch = epoch + 1
                print(f"New best accuracy: {best_accuracy:.4f} - Saving model...")

                model.module.save_pretrained(CONFIG["paths"]["output_dir"])
                tokenizer.save_pretrained(CONFIG["paths"]["output_dir"])

                if CONFIG["training"]["save_training_state"]:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_accuracy': best_accuracy,
                    }, os.path.join(CONFIG["paths"]["output_dir"], 'training_state.pt'))

                # Save evaluation for every best checkpoint in case the training crashed midrun
                duration_df, hours, minutes, seconds = get_duration_df(start_time, best_epoch)
                excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'classification_reports.xlsx')
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    best_report_df.to_excel(writer, sheet_name='best_validation_report')
                    val_predictions_df.to_excel(writer, sheet_name="val_predictions")
                    duration_df.to_excel(writer, sheet_name='duration', index=False)

    # Update the test evaluation part in the train function
    if rank == 0:
        print("\nEvaluate best model on test data")

    test_df = pd.read_csv(os.path.join(CONFIG["paths"]["data_dir"], "test.csv"))
    test_loss, test_accuracy, test_report_df, test_predictions_df = evaluate(
        best_model,
        test_dataloader,
        device,
        labels=labels_sorted, # Pass label IDs
        target_names=target_names, # Pass label names
        is_test=True,
        test_df=test_df,  # Pass the test DataFrame
        id2label=id2label
    )

    if rank == 0:
        print("\nTest Classification Report:")
        print(test_report_df)

        # Save all reports to a single Excel file
        duration_df, hours, minutes, seconds = get_duration_df(start_time, best_epoch)
        excel_path = os.path.join(CONFIG["paths"]["output_dir"], 'classification_reports.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            best_report_df.to_excel(writer, sheet_name='best_validation_report')
            test_report_df.to_excel(writer, sheet_name='test_report')
            test_predictions_df.to_excel(writer, sheet_name='test_predictions', index=False)
            duration_df.to_excel(writer, sheet_name='duration', index=False)

        print(f"\nAll reports saved to {excel_path}")
        print(f"Training duration - {hours}h {minutes}m {seconds}s")

        test_eval_dict = {
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "training_duration_hours": hours,
            "training_duration_minutes": minutes,
            "training_duration_seconds": seconds
        }
        wandb.log(test_eval_dict)
        test_eval_json_path = os.path.join(CONFIG["paths"]["output_dir"], 'test_evaluation.json')
        with open(test_eval_json_path, "w+") as f:
            f.write(json.dumps(test_eval_dict, indent=3))

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
