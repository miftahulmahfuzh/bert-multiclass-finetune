import csv
import json
import logging
import math
import random
import shutil
import sys
from copy import deepcopy
from os.path import basename
from pathlib import Path
from typing import Any, Optional, cast

import datasets
import evaluate
import mlflow
import nltk
import numpy as np
import torch
from datasets import Dataset as HDataset
from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from prosa_nlp.utils import dataset_fingerprint, fix_punctuation

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

metric = evaluate.load("./prosa_nlp/metrics/rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def get_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}  # type: ignore
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    return compute_metrics


def get_preprocess_function(
    model_name,
    tokenizer,
    max_input_length,
    max_target_length,
    prefix,
    source,
    target,
):
    # PADDING:
    # "Whether to pad all samples to model maximum sentence length. "
    # "If False, will pad the samples dynamically when batching to the maximum
    #    length in the batch
    # Mode "efficient on GPU but very bad for TPU."
    # `False` or `"max_length"`

    padding = False
    ignore_pad_token_for_loss = True

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[source])):
            if examples[source][i] and examples[target][i]:
                inputs.append(examples[source][i])
                targets.append(examples[target][i])

        skip_pad = [893, 117, 97] # this is mod 1000 of total rows in train, dev, test
        if len(inputs) not in skip_pad:
            for k in range(1000 - len(inputs)):
                inputs.append(inputs[-1])
                targets.append(targets[-1])
        # print(f"LEN INPUTS {len(inputs)}")
        # assert len(inputs) == len(targets), f"LENGTH IS NOT THE SAME. INPUTS {len(inputs)}. TARGETS {len(targets)}"

        inputs = [prefix + inp for inp in inputs]
        # inputs, max_length=max_target_length, padding=padding, truncation=True
        model_inputs = tokenizer(
            inputs, max_length=480, padding=False, truncation=True
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            # max_length=max_target_length,
            max_length=32,
            # padding=padding,
            padding=False,
            truncation=True
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels
        # by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [
                    (label if label != tokenizer.pad_token_id else -100)
                    for label in label
                ]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # for k, items in model_inputs.items():
        #     print(f"LEN FIELD {k}: {len(items)}")
        return model_inputs

    return preprocess_function


def train(
    pretrained_model: str,
    data: str,
    data_kwargs: dict[str, Any],
    max_input_length: int,
    max_target_length: int,
    source: str,
    target: str,
    prefix: str,
    batch_size: int,
    epoch: int,
    learning_rate: float,
    ort: bool,
    from_flax: bool,
    num_validation_examples: Optional[int] = None,
    num_test_examples: Optional[int] = None,
    max_train_examples: int = 0,
    dataset_processor: None | str = None,
    config_file: Path | None = None,
    params: dict | None = None,
):
    if "names" in data_kwargs:
        nn = "+".join(data_kwargs["names"])
        print("USING INDONLG for", data_kwargs["names"], pretrained_model)
    elif "train_names" in data_kwargs or "test_names" in data_kwargs:
        if "train_names" not in data_kwargs or "test_names" not in data_kwargs:
            print("'train_names' and 'test_names' must be specified")
            return
        nn = "+".join(data_kwargs["train_names"])
        nn += "_" + ("+".join(data_kwargs["test_names"]))
    elif "name" in data_kwargs:
        nn = data_kwargs["name"]
        print("USING INDONLG for", data_kwargs["name"], pretrained_model)
    else:
        nn = "miftah"

    print("NN", nn)
    print("* loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    print("* loading dataset")

    if "names" in data_kwargs:
        print("With names")
        names = data_kwargs["names"]
        del data_kwargs["names"]
        d = []
        for name in names:
            # dataset = load_dataset(data, trust_remote_code=True, name=name, **data_kwargs)
            dataset = load_dataset(data, name=name, **data_kwargs)
            d.append(dataset)

        dataset = DatasetDict(
            {
                "train": concatenate_datasets([data["train"] for data in d]),
                "test": concatenate_datasets([data["test"] for data in d]),
                "validation": concatenate_datasets([data["validation"] for data in d]),
            }
        )
    elif "train_names" in data_kwargs:
        train_names = data_kwargs["train_names"]
        del data_kwargs["train_names"]
        d_train = []
        test_names = data_kwargs["test_names"]
        del data_kwargs["test_names"]
        d_test = []

        for name in train_names:
            # dataset = load_dataset(data, trust_remote_code=True, name=name, **data_kwargs)
            dataset = load_dataset(data, name=name, **data_kwargs)
            d_train.append(dataset)

        for name in test_names:
            # dataset = load_dataset(data, trust_remote_code=True, name=name, **data_kwargs)
            dataset = load_dataset(data, name=name, **data_kwargs)
            d_test.append(dataset)
        dataset = DatasetDict(
            {
                "train": concatenate_datasets([data["train"] for data in d_train]),
                "test": concatenate_datasets([data["test"] for data in d_test]),
                "validation": concatenate_datasets(
                    [data["validation"] for data in d_train]
                ),
            }
        )

    else:
        # dataset = load_dataset(data, trust_remote_code=True, **data_kwargs)
        dataset = load_dataset(data, **data_kwargs)
    dataset = cast(DatasetDict, dataset)

    if max_train_examples > 0:
        dataset["train"] = HDataset.from_dict(dataset["train"][0:max_train_examples])

    # dataset["train"] = HDataset.from_dict(dataset["train"][0:300])
    # dataset["validation"] = HDataset.from_dict(dataset["validation"][0:50])
    # dataset["test"] = HDataset.from_dict(dataset["test"][0:5])

    # cut = 0
    # if "test" not in dataset:
    #     if not num_test_examples:
    #         raise Exception("Must provide num_validation_examples")
    #     dataset["test"] = HDataset.from_dict(
    #         dataset["train"][cut : cut + num_test_examples]
    #     )
    #     cut += num_test_examples
    #
    # if "validation" not in dataset:
    #     if not num_validation_examples:
    #         raise Exception("Must provide num_validation_examples")
    #     dataset["validation"] = HDataset.from_dict(
    #         dataset["train"][0:num_validation_examples]
    #     )
    #     cut += num_validation_examples
    #
    # print("Train Offset", cut)
    # dataset["train"] = HDataset.from_dict(dataset["train"][cut:])

    fingerprint = dataset_fingerprint(dataset)
    preprocess_function = get_preprocess_function(
        pretrained_model,
        tokenizer,
        max_input_length,
        max_target_length,
        prefix,
        source,
        target,
    )

    print("* DATASET", dataset)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)

    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model, from_flax=from_flax)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )
    print("* Model loaded")

    Path("models/seq2seq").mkdir(parents=True, exist_ok=True)
    if ort:
        output_dir = Path("models/seq2seq").joinpath(
            f"{basename(data)}-{basename(pretrained_model.rstrip('/'))}-{nn}-{epoch}-ort"
        )
    else:
        output_dir = Path("models/seq2seq").joinpath(
            f"{basename(data)}-{basename(pretrained_model.rstrip('/'))}-{nn}-{epoch}"
        )

    if max_train_examples > 0:
        output_dir = output_dir.with_name(output_dir.name + f"-{max_train_examples}")

    print("OUT DIR", output_dir)

    # https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8
    weight_decay = 0.05 * math.sqrt(batch_size / (len(dataset["train"]) * epoch))

    if nn:
        experiment = data.replace("/", "--") + f"-{nn}"

    print("EXPERIMENT", experiment)
    mlflow.set_experiment(experiment)

    if params:
        params["max_train_examples"] = len(dataset["train"])
        mlflow.log_params(params)
    mlflow.log_param(key="dataset_fingerprint", value=fingerprint)
    mlflow.log_param(key="dataset_processor", value=dataset_processor)
    mlflow.log_param(key="weight_decay", value=weight_decay)
    mlflow.log_param(key="device_count", value=torch.cuda.device_count())

    print(f"** Weight Decay: {weight_decay}")
    print(f"** Prefix      : '{prefix}'")
    print(f"** Device Count: {torch.cuda.device_count()}")
    if ort:
        from optimum.onnxruntime import ORTSeq2SeqTrainer, ORTSeq2SeqTrainingArguments

        training_args = ORTSeq2SeqTrainingArguments(
            output_dir=str(output_dir),
            learning_rate=learning_rate,
            # generation_max_length=max_target_length,
            predict_with_generate=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            evaluation_strategy="epoch",
            weight_decay=weight_decay,
            save_total_limit=2,
            save_strategy="no",
            save_safetensors=False,
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to="mlflow",  # type: ignore
        )
        trainer = ORTSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],  # type: ignore
            eval_dataset=tokenized_datasets["validation"],  # type: ignore
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=get_compute_metrics(tokenizer),
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            learning_rate=learning_rate,
            # generation_max_length=max_target_length,
            predict_with_generate=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            evaluation_strategy="epoch",
            weight_decay=weight_decay,
            save_total_limit=2,
            save_strategy="no",
            save_safetensors=False,
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to="mlflow",  # type: ignore
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],  # type: ignore
            eval_dataset=tokenized_datasets["validation"],  # type: ignore
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=get_compute_metrics(tokenizer),
        )

    trainer.train()
    trainer.save_model()

    print("\n====EVALUATE=====")

    input_lengths = [
        len(example) for example in tokenized_datasets["test"]["input_ids"]
    ]
    tokenized_datasets["test"] = tokenized_datasets["test"].add_column(  # type: ignore
        "input_length", input_lengths
    )

    tokenized_datasets["test"] = tokenized_datasets["test"].sort("input_length")

    datasets.disable_progress_bar()
    N = 64
    preds = []
    labels = []
    sources = []
    for i in range(0, len(tokenized_datasets["test"]), N):
        print(i + 1)
        batch = HDataset.from_dict(tokenized_datasets["test"][i : i + N])
        batch_raw = dataset["test"][i : i + N] # miftah
        input_len = max([len(example) for example in batch["input_ids"]])
        if input_len < 10:
            max_length = 20
        else:
            max_length = math.floor(input_len * 1.5)

        batch_preds, batch_labels, _ = trainer.predict(
            batch,
            max_length=max_length,  # type: ignore
        )
        batch_preds = np.where(batch_preds != -100, batch_preds, tokenizer.pad_token_id)  # type: ignore
        try:
            decoded_preds = tokenizer.batch_decode(
                batch_preds, skip_special_tokens=True
            )
        except Exception:
            print(batch_preds)
            print(i + 1)
            sys.exit(1)
            # continue
        # decoded_labels = tokenized_datasets["test"][target][i : i + N]
        decoded_labels = dataset["test"][target][i : i + N]
        preds.extend([fix_punctuation(pred) for pred in decoded_preds])
        labels.extend([fix_punctuation(label) for label in decoded_labels])
        sources.extend(batch_raw[source])

    eval_path = output_dir.joinpath("eval")
    eval_path.mkdir(exist_ok=True)

    config_path = output_dir.joinpath("config")
    if config_file:
        config_path.mkdir(exist_ok=True)
        shutil.copy(config_file, config_path)

    exact_match = 0
    exact_match_lower = 0

    header = [source, target, f"{target}_predicted"]

    with eval_path.joinpath("predictions.csv").open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for source, pred, label in zip(sources, preds, labels):
            # FIX: Unicode error: ????
            source = source.replace(",", " ")
            writer.writerow((source, label, pred))

    with eval_path.joinpath("correct_predictions.csv").open(
        "w"
    ) as f, eval_path.joinpath("correct_predictions_lower.csv").open("w") as f_lower:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer_lower = csv.writer(f_lower, lineterminator="\n")
        writer_lower.writerow(header)
        for source, pred, label in zip(sources, preds, labels):
            source = source.replace(",", " ")
            if label == pred:
                exact_match += 1
                writer.writerow((source, label, pred))
            if label.lower() == pred.lower():
                exact_match_lower += 1
                writer_lower.writerow((source, label, pred))

    with eval_path.joinpath("wrong_predictions.csv").open("w") as f, eval_path.joinpath(
        "wrong_predictions_lower.csv"
    ).open("w") as f_lower:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer_lower = csv.writer(f_lower, lineterminator="\n")
        writer_lower.writerow(header)
        for source, pred, label in zip(sources, preds, labels):
            source = source.replace(",", " ")
            if label != pred:
                writer.writerow((source, label, pred))
            if label.lower() != pred.lower():
                writer_lower.writerow((source, label, pred))

    rouges: dict[str, float] = evaluate.load("./prosa_nlp/metrics/rouge").compute(  # type: ignore
        predictions=preds, references=labels
    )
    bleus: dict[str, float | list[float]] = evaluate.load(
        "./prosa_nlp/metrics/bleu"
    ).compute(predictions=preds, references=labels)  # type: ignore
    rouges_lower: dict[str, float] = evaluate.load("./prosa_nlp/metrics/rouge").compute(
        predictions=[pred.lower() for pred in preds],
        references=[label.lower() for label in labels],
    )  # type: ignore
    bleus_lower: dict[str, float | list[float]] = evaluate.load(
        "./prosa_nlp/metrics/bleu"
    ).compute(
        predictions=[pred.lower() for pred in preds],
        references=[label.lower() for label in labels],
    )  # type: ignore

    metrics = {
        "rouge1": rouges["rouge1"] * 100,
        "rouge2": rouges["rouge2"] * 100,
        "rougeL": rouges["rougeL"] * 100,
        "rougeLsum": rouges["rougeLsum"] * 100,
        "bleu": bleus["bleu"] * 100,
        "bleu_precisions": [b * 100 for b in bleus["precisions"]],
        "rouge1_lower": rouges_lower["rouge1"] * 100,
        "rouge2_lower": rouges_lower["rouge2"] * 100,
        "rougeL_lower": rouges_lower["rougeL"] * 100,
        "rougeLsum_lower": rouges_lower["rougeLsum"] * 100,
        "bleu_lower": bleus_lower["bleu"] * 100,
        "bleu_precisions_lower": [b * 100 for b in bleus_lower["precisions"]],
        "num_test_examples": len(sources),
        "exact_match_acc": exact_match / len(sources),
        "exact_match_acc_lower": exact_match_lower / len(sources),
    }

    with eval_path.joinpath("result.json").open("w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))

    mlflow.log_metric(key="bleu", value=metrics["bleu"])  # type: ignore
    mlflow.log_metric(key="rouge1", value=metrics["rouge1"])  # type: ignore
    mlflow.log_metric(key="rouge2", value=metrics["rouge2"])  # type: ignore
    mlflow.log_metric(key="rougeL", value=metrics["rougeL"])  # type: ignore

    if not dataset_processor:
        dataset_processor = ""

    mlflow.log_artifact(str(eval_path.joinpath("wrong_predictions.csv")))
    mlflow.log_artifact(str(eval_path.joinpath("result.json")))


def eval(
    model_name: str,
    data: str,
    data_kwargs: dict[str, Any],
    max_input_length: int,
    max_target_length: int,
    source: str,
    target: str,
    prefix: str,
):
    from transformers import pipeline

    output_dir = Path(model_name)

    print("* loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    model = pipeline("text2text-generation", model=model_name, device=0)

    print("* loading dataset")
    # dataset = load_dataset(data, trust_remote_code=True, **data_kwargs)
    dataset = load_dataset(data, **data_kwargs)
    dataset = cast(DatasetDict, dataset)
    # dataset["test"] = HDataset.from_dict(dataset["test"][2245:2255])

    preprocess_function = get_preprocess_function(
        model_name,
        tokenizer,
        max_input_length,
        max_target_length,
        prefix,
        source,
        target,
    )

    print("* DATASET", dataset)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(dataset["test"].column_names)
    print("\n====EVALUATE=====")

    input_lengths = [
        len(example) for example in tokenized_datasets["test"]["input_ids"]
    ]
    tokenized_datasets["test"] = tokenized_datasets["test"].add_column(  # type: ignore
        "input_length", input_lengths
    )
    tokenized_datasets["test"] = tokenized_datasets["test"].sort("input_length")

    datasets.disable_progress_bar()
    N = 64
    preds = []
    labels = []
    sources = []
    for i in range(0, len(tokenized_datasets["test"]), N):
        print(i + 1)
        batch = HDataset.from_dict(tokenized_datasets["test"][i : i + N])
        input_len = max([len(example) for example in batch["input_ids"]])
        if input_len < 10:
            max_length = 20
        else:
            max_length = math.floor(input_len * 1.5)

        batch_preds = model(
            batch["text"],
            max_length=max_length,  # type: ignore
        )
        try:
            decoded_preds = [pred["generated_text"] for pred in batch_preds]
        except Exception:
            print(batch_preds)
            print(i + 1)
            sys.exit(1)
            # continue
        decoded_labels = tokenized_datasets["test"][target][i : i + N]
        preds.extend([fix_punctuation(pred) for pred in decoded_preds])
        labels.extend([fix_punctuation(label) for label in decoded_labels])
        sources.extend(batch[source])

    eval_path = output_dir.joinpath("eval")
    eval_path.mkdir(exist_ok=True)

    exact_match = 0
    exact_match_lower = 0

    header = [source, target, f"{target}_predicted"]

    with eval_path.joinpath("predictions.csv").open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for source, pred, label in zip(sources, preds, labels):
            # FIX: Unicode error: ????
            try:
                writer.writerow((source, label, pred))
            except Exception as e:
                print(
                    "==================================================================="
                )
                print(e)
                print(
                    "*******************************************************************"
                )

    with eval_path.joinpath("correct_predictions.csv").open(
        "w"
    ) as f, eval_path.joinpath("correct_predictions_lower.csv").open("w") as f_lower:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer_lower = csv.writer(f_lower, lineterminator="\n")
        writer_lower.writerow(header)
        for source, pred, label in zip(sources, preds, labels):
            if label == pred:
                exact_match += 1
                try:
                    writer.writerow((source, label, pred))
                except Exception as e:
                    print(
                        "==================================================================="
                    )
                    print(e)
                    print(
                        "*******************************************************************"
                    )
            if label.lower() == pred.lower():
                exact_match_lower += 1
                try:
                    writer_lower.writerow((source, label, pred))
                except Exception as e:
                    print(
                        "==================================================================="
                    )
                    print(e)
                    print(
                        "*******************************************************************"
                    )

    with eval_path.joinpath("wrong_predictions.csv").open("w") as f, eval_path.joinpath(
        "wrong_predictions_lower.csv"
    ).open("w") as f_lower:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer_lower = csv.writer(f_lower, lineterminator="\n")
        writer_lower.writerow(header)
        for source, pred, label in zip(sources, preds, labels):
            if label != pred:
                try:
                    writer.writerow((source, label, pred))
                except Exception as e:
                    print(
                        "==================================================================="
                    )
                    print(e)
                    print(
                        "*******************************************************************"
                    )
            if label.lower() != pred.lower():
                try:
                    writer_lower.writerow((source, label, pred))
                except Exception as e:
                    print(
                        "==================================================================="
                    )
                    print(e)
                    print(
                        "*******************************************************************"
                    )

    rouges: dict[str, float] = evaluate.load("./prosa_nlp/metrics/rouge").compute(  # type: ignore
        predictions=preds, references=labels
    )
    bleus: dict[str, float | list[float]] = evaluate.load(
        "./prosa_nlp/metrics/bleu"
    ).compute(predictions=preds, references=labels)  # type: ignore
    rouges_lower: dict[str, float] = evaluate.load("./prosa_nlp/metrics/rouge").compute(
        predictions=[pred.lower() for pred in preds],
        references=[label.lower() for label in labels],
    )  # type: ignore
    bleus_lower: dict[str, float | list[float]] = evaluate.load(
        "./prosa_nlp/metrics/bleu"
    ).compute(
        predictions=[pred.lower() for pred in preds],
        references=[label.lower() for label in labels],
    )  # type: ignore

    metrics = {
        "rouge1": rouges["rouge1"] * 100,
        "rouge2": rouges["rouge2"] * 100,
        "rougeL": rouges["rougeL"] * 100,
        "rougeLsum": rouges["rougeLsum"] * 100,
        "bleu": bleus["bleu"] * 100,
        "bleu_precisions": [b * 100 for b in bleus["precisions"]],
        "rouge1_lower": rouges_lower["rouge1"] * 100,
        "rouge2_lower": rouges_lower["rouge2"] * 100,
        "rougeL_lower": rouges_lower["rougeL"] * 100,
        "rougeLsum_lower": rouges_lower["rougeLsum"] * 100,
        "bleu_lower": bleus_lower["bleu"] * 100,
        "bleu_precisions_lower": [b * 100 for b in bleus_lower["precisions"]],
        "num_test_examples": len(sources),
        "exact_match_acc": exact_match / len(sources),
        "exact_match_acc_lower": exact_match_lower / len(sources),
    }

    with eval_path.joinpath("result.json").open("w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
