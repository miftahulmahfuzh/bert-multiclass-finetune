# Comment Generation Model Finetuning

This repository contains code for fine-tuning a large language model (LLM) for the task of comment generation based on input posts. It leverages Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters, and 4-bit quantization for efficient training and inference.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training Mode](#training-mode)
  - [Test Mode](#test-mode)
- [Caching](#caching)
- [Model and Tokenizer](#model-and-tokenizer)
- [Data](#data)
- [Logging and Monitoring](#logging-and-monitoring)
- [Output](#output)
- [Notes](#notes)

## Overview

This project fine-tunes a pretrained causal language model to generate comments given input posts. It uses:

- Hugging Face `transformers` for model and tokenizer loading
- `peft` library for LoRA adapters to enable parameter-efficient fine-tuning
- 4-bit quantization via `bitsandbytes` for memory-efficient training
- `datasets` library for loading and splitting datasets
- `wandb` for experiment tracking and logging

## Features

- Supports training and test-only modes
- Caching of tokenized datasets to speed up repeated runs
- Custom dataset class with prompt-based tokenization
- Detailed prediction evaluation with output saved to Excel
- LoRA adapter integration for efficient fine-tuning
- Configurable training parameters via JSON config

## Requirements

The required Python packages are listed in `requirements.txt`. Key dependencies include:

- torch
- transformers
- datasets
- peft
- bitsandbytes
- wandb
- pandas

Make sure you have a CUDA-enabled GPU for best performance.

## Setup

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset or use the provided dataset name in the config.
4. Prepare your prompt template file (e.g., `prompts/prompt_v1.txt`).

## Configuration

The main configuration file is `finetune_config.json`. It contains sections for:

- `model`: model name to load from Hugging Face Hub
- `paths`: data directory, cache usage, prompt file, output directory, and adapter paths
- `data`: dataset name, validation split ratio, input/output column names
- `training`: mode (`train` or `test`), batch size, epochs, learning rate, max sequence length, logging and saving steps, etc.
- `lora`: LoRA adapter parameters
- `wandb`: Weights & Biases project name

Modify this file to suit your environment and training needs.

## Usage

Run the main training script:

```bash
python llm_train_v5.py
```

### Training Mode

- The default mode is `train`.
- The script loads the pretrained model with 4-bit quantization.
- It prepares the dataset with caching and tokenization.
- LoRA adapters are applied for efficient fine-tuning.
- Training progress and metrics are logged to Weights & Biases.
- After training, detailed predictions on the test set are generated and saved.

### Test Mode

- Set `"mode": "test"` in the config.
- The script loads the trained LoRA adapter from the specified path.
- It generates predictions on the test set without further training.
- Results are saved and logged to Weights & Biases.

## Caching

Tokenized datasets are cached in the `data/cache` directory to speed up repeated runs. The cache filename is based on a hash of the model name, max length, and split name.

## Model and Tokenizer

- The model is loaded from Hugging Face Hub with 4-bit quantization for memory efficiency.
- The tokenizer is loaded and padded with the EOS token if no pad token is defined.
- LoRA adapters are applied on top of the base model.

## Data

- The dataset is loaded using the Hugging Face `datasets` library.
- If no validation or test split is found, the script automatically splits the training data.
- The dataset columns for input and output are configurable.
- A prompt template is used to format the input text.

## Logging and Monitoring

- Training metrics and evaluation results are logged to Weights & Biases (wandb).
- The wandb project and run names are configurable.

## Output

- Model checkpoints and best model are saved in the output directory with timestamped folder names.
- Detailed test predictions are saved as an Excel file (`test_predictions.xlsx`) in the output directory.
- The training configuration is saved alongside outputs.

## Notes

- The script increases the system file descriptor limit to handle large datasets.
- The training uses mixed precision (fp16) for faster training on GPUs.
- The generation during evaluation uses sampling with temperature, top-k, top-p, and repetition penalty for diverse outputs.

---

For any questions or issues, please open an issue or contact the maintainer.

