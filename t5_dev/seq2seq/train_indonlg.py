import csv
import json
import math
import pickle
import random
import shutil
from hashlib import md5
from os.path import basename
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import datasets
import evaluate
import mlflow
import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch import optim
from transformers import GPT2LMHeadModel, MBartForConditionalGeneration

from prosa_nlp.indonlg.modules.tokenization_indonlg import IndoNLGTokenizer
from prosa_nlp.indonlg.utils.data_utils import (
    GenerationDataLoader,
    MachineTranslationDataset,
)
from prosa_nlp.indonlg.utils.forward_fn import forward_generation
from prosa_nlp.indonlg.utils.metrics import generation_metrics_fn
from prosa_nlp.indonlg.utils.train_eval import train as train_model
from prosa_nlp.utils import fix_punctuation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(0)


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
    num_validation_examples: Optional[int] = None,
    num_test_examples: Optional[int] = None,
    max_train_examples: int = 0,
    dataset_processor: None | str = None,
    config_file: Path | None = None,
    params: dict | None = None,
):
    print("============================")

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
        nn = ""
    print("NN", nn)

    tokenizer = IndoNLGTokenizer.from_pretrained(pretrained_model)

    if "names" in data_kwargs:
        names = data_kwargs["names"]
        del data_kwargs["names"]
        d = []
        for name in names:
            dataset = load_dataset(
                data, trust_remote_code=True, name=name, **data_kwargs
            )
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
            dataset = load_dataset(
                data, trust_remote_code=True, name=name, **data_kwargs
            )
            d_train.append(dataset)

        for name in test_names:
            dataset = load_dataset(
                data, trust_remote_code=True, name=name, **data_kwargs
            )
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
        dataset = load_dataset(data, trust_remote_code=True, **data_kwargs)
    dataset = cast(DatasetDict, dataset)

    if max_train_examples > 0:
        dataset["train"] = Dataset.from_dict(dataset["train"][0:max_train_examples])
    # dataset["train"] = Dataset.from_dict(dataset["train"][0:300])

    # cut = 0
    # if "test" not in dataset:
    #     if not num_test_examples:
    #         raise Exception("Must provide num_validation_examples")
    #     dataset["test"] = Dataset.from_dict(
    #         dataset["train"][cut : cut + num_test_examples]
    #     )
    #     cut += num_test_examples
    #
    # if "validation" not in dataset:
    #     if not num_validation_examples:
    #         raise Exception("Must provide num_validation_examples")
    #     dataset["validation"] = Dataset.from_dict(
    #         dataset["train"][0:num_validation_examples]
    #     )
    #     cut += num_validation_examples
    #
    # dataset["train"] = Dataset.from_dict(dataset["train"][cut:])

    # dataset["train"] = Dataset.from_dict(dataset["train"][0:100])
    # dataset["validation"] = Dataset.from_dict(dataset["validation"][0:10])
    # dataset["test"] = Dataset.from_dict(dataset["test"][1000:3000])

    print("* DATASET", dataset)
    if "indobart" in pretrained_model:
        model = MBartForConditionalGeneration.from_pretrained(pretrained_model)
        model = cast(MBartForConditionalGeneration, model)
    elif "indogpt" in pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        model = cast(GPT2LMHeadModel, model)

    model = model.cuda()  # type: ignore
    tokenizer = IndoNLGTokenizer.from_pretrained(pretrained_model)

    cache_dir = (
        Path(".cache")
        / "dataset"
        / (
            data.replace("/", "--")
            + "-"
            + md5(pickle.dumps(dataset)).hexdigest()[:6]
            + "-indonlg"
        )
    )
    print("* Using cache dir:", cache_dir)

    if not cache_dir.exists():
        print("* Cache miss")
        cache_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "validation", "test"]:
            print(f"* Generate json for {split} data")
            examples = []
            for i, example in enumerate(dataset[split]):
                example = cast(Mapping[str, str], example)
                examples.append(
                    {
                        "id": f"{split}-{i+1}",
                        "text": example[source],
                        "label": example[target],
                    }
                )

            with (cache_dir / f"{split}.json").open("w") as f:
                json.dump(examples, f, indent=2)
    else:
        print("* Cache hit")

    lr = learning_rate
    gamma = 0.9
    lower = False
    step_size = 1
    beam_size = 5
    max_norm = 10
    early_stop = 5

    max_seq_len = 512
    grad_accumulate = 1

    if "indobart" in pretrained_model:
        model_type = "indo-bart"
        max_seq_len = 1024
    elif "indogpt" in pretrained_model:
        model_type = "indo-gpt2"

    print("Model Type:", model_type)
    valid_criterion = "SacreBLEU"

    train_batch_size = batch_size
    valid_batch_size = batch_size
    test_batch_size = batch_size

    optimizer = optim.Adam(model.parameters(), lr=lr)
    source_lang = "[indonesian]"
    target_lang = "[indonesian]"
    src_lid = tokenizer.special_tokens_to_ids[source_lang]
    tgt_lid = tokenizer.special_tokens_to_ids[target_lang]

    model.config.decoder_start_token_id = tgt_lid

    Path("models/seq2seq").mkdir(parents=True, exist_ok=True)

    if ort:
        output_dir = Path("models/seq2seq").joinpath(
            f"{basename(data)}-{basename(pretrained_model.rstrip('/'))}-{nn}-{epoch}-ort"
        )
    else:
        output_dir = Path("models/seq2seq").joinpath(
            f"{basename(data)}-{basename(pretrained_model.rstrip('/'))}-{nn}-{epoch}"
        )
    print("Output dir:", output_dir)

    device = "cuda0"
    # set a specific cuda device
    if "cuda" in device:
        torch.cuda.set_device(int(device[4:]))
        device = "cuda"
        model = model.cuda()

    train_dataset_path = str(cache_dir / "train.json")
    valid_dataset_path = str(cache_dir / "validation.json")
    test_dataset_path = str(cache_dir / "test.json")

    train_dataset = MachineTranslationDataset(
        train_dataset_path,
        tokenizer,
        lowercase=lower,
        swap_source_target=False,
    )
    valid_dataset = MachineTranslationDataset(
        valid_dataset_path,
        tokenizer,
        lowercase=lower,
        swap_source_target=False,
    )
    test_dataset = MachineTranslationDataset(
        test_dataset_path,
        tokenizer,
        lowercase=lower,
        swap_source_target=False,
    )

    train_loader = GenerationDataLoader(
        dataset=train_dataset,
        model_type=model_type,  # type: ignore
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=train_batch_size,
        src_lid_token_id=src_lid,
        tgt_lid_token_id=tgt_lid,
        num_workers=8,
        shuffle=True,
    )
    valid_loader = GenerationDataLoader(
        dataset=valid_dataset,
        model_type=model_type,  # type: ignore
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=valid_batch_size,
        src_lid_token_id=src_lid,
        tgt_lid_token_id=tgt_lid,
        num_workers=8,
        shuffle=False,
    )

    test_loader = GenerationDataLoader(
        dataset=test_dataset,
        model_type=model_type,  # type: ignore
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=test_batch_size,
        src_lid_token_id=src_lid,
        tgt_lid_token_id=tgt_lid,
        num_workers=8,
        shuffle=False,
    )
    n_epochs = epoch

    if nn:
        experiment = data.replace("/", "--") + f"-{nn}"
    print("EXPERIMENT", experiment)
    mlflow.set_experiment(experiment)

    if params:
        mlflow.log_params(params)
    mlflow.log_param(key="dataset_processor", value=dataset_processor)
    mlflow.log_param(key="device_count", value=torch.cuda.device_count())

    train_model(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        forward_fn=forward_generation,
        metrics_fn=generation_metrics_fn,
        valid_criterion=valid_criterion,
        tokenizer=tokenizer,
        n_epochs=n_epochs,
        evaluate_every=1,
        early_stop=early_stop,
        grad_accum=grad_accumulate,
        step_size=step_size,
        gamma=gamma,
        max_norm=max_norm,
        model_type=model_type,  # type: ignore
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        model_dir=str(output_dir),
        exp_id=0,
        fp16=False,
        device=device,
    )
    tokenizer.save_pretrained(str(output_dir))

    tokenizer = IndoNLGTokenizer.from_pretrained(str(output_dir))

    if "indobart" in pretrained_model:
        model = MBartForConditionalGeneration.from_pretrained(output_dir)
        model = cast(MBartForConditionalGeneration, model)
        tokenizer_model_type = "indobart"
    elif "indogpt" in pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        model = cast(GPT2LMHeadModel, model)
        tokenizer_model_type = "indogpt"

    device = "cuda0"
    # set a specific cuda device
    if "cuda" in device:
        torch.cuda.set_device(int(device[4:]))
        device = "cuda"
        model = model.cuda()

    input_lengths = [len(example) for example in dataset["test"][source]]
    dataset["test"] = dataset["test"].add_column("input_length", input_lengths)  # type: ignore
    dataset["test"] = dataset["test"].sort("input_length")

    preds = []
    labels = []
    sources = []

    # from IPython import embed
    #
    # embed()

    datasets.disable_progress_bar()
    tests = list(test_loader)
    for i, batch in enumerate(tests):
        print(f"{i + 1}/{len(tests)}")
        _, pred, label = forward_generation(
            model,
            batch,
            tokenizer,
            model_type,
            is_inference=True,
            device="cuda",
            is_test=True,
            repetition_penalty=1.25,
        )
        preds.extend(pred)
        labels.extend([l.strip() for l in label])

        start = i * test_batch_size
        end = start + test_batch_size
        sources.extend([d["text"] for d in test_dataset.get(start, end)])

    # for i in range(0, len(dataset["test"]), N):
    #     print(i + 1)
    #     batch = Dataset.from_dict(dataset["test"][i : i + N])
    #     batch_tokenized = tokenizer.prepare_input_for_generation(
    #         batch[source],
    #         return_tensors="pt",
    #         model_type=tokenizer_model_type,  # type: ignore
    #     ).to("cuda")
    #     input_len = batch_tokenized.input_ids.size(1)
    #
    #     max_length = 0
    #     if "indobart" in pretrained_model:
    #         if input_len < 10:
    #             max_length = 25
    #         else:
    #             max_length = math.floor(input_len * 1.5) + 5
    #     elif "indogpt" in pretrained_model:
    #         max_length = max_seq_len * 2
    #
    #     print("XXX", max_length, max_seq_len)
    #     if max_length > max_seq_len:
    #         max_length = max_seq_len
    #
    #     batch_preds = model.generate(**batch_tokenized, max_length=max_length)  # type: ignore
    #     decoded_preds = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
    #
    #     decoded_preds = [d.strip() for d in decoded_preds]
    #     decoded_labels = dataset["test"][target][i : i + N]
    #
    #     if lower:
    #         decoded_preds = [d.lower() for d in decoded_preds]
    #         decoded_labels = [d.lower() for d in decoded_labels]
    #
    #     preds.extend([fix_punctuation(pred) for pred in decoded_preds])
    #     labels.extend([fix_punctuation(label) for label in decoded_labels])
    #     sources.extend(batch[source])

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
            writer.writerow((source, label, pred))

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
            if label != pred:
                writer.writerow((source, label, pred))
            if label.lower() != pred.lower():
                writer_lower.writerow((source, label, pred))

    rouges: dict[str, float] = evaluate.load("rouge").compute(
        predictions=preds, references=labels
    )  # type: ignore
    bleus: dict[str, float | list[float]] = evaluate.load("bleu").compute(
        predictions=preds, references=labels
    )  # type: ignore
    rouges_lower: dict[str, float] = evaluate.load("rouge").compute(
        predictions=[pred.lower() for pred in preds],
        references=[label.lower() for label in labels],
    )  # type: ignore
    bleus_lower: dict[str, float | list[float]] = evaluate.load("bleu").compute(
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
