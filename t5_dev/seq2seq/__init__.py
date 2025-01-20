import tempfile
from functools import partial
from pathlib import Path
from time import perf_counter as pc
from typing import cast

import yaml
from typer import Typer

app = Typer()


@app.command("ft")
def fine_tune(config_file: Path):
    import mlflow

    from .config import Config

    with config_file.open() as f:
        data = yaml.load(f, yaml.Loader)
        conf = Config.parse_obj(data)

    if "indobenchmark" in conf.pretrained_model:
        from .train_indonlg import train as train_indonlg

        train_indonlg(
            pretrained_model=conf.pretrained_model,
            data=conf.data,
            data_kwargs=conf.data_kwargs,
            max_input_length=conf.max_input_length,
            max_target_length=conf.max_target_length,
            source=conf.source,
            target=conf.target,
            prefix=conf.prefix,
            batch_size=conf.batch_size,
            epoch=conf.training.epoch,
            ort=conf.ort,
            num_test_examples=conf.num_test_examples,
            num_validation_examples=conf.num_validation_examples,
            max_train_examples=conf.max_train_examples,
            dataset_processor=conf.dataset_processor,
            learning_rate=conf.training.lr,
            config_file=config_file,
            params=conf.params(),
        )
    else:
        from .train import train

        train(
            pretrained_model=conf.pretrained_model,
            data=conf.data,
            data_kwargs=conf.data_kwargs,
            max_input_length=conf.max_input_length,
            max_target_length=conf.max_target_length,
            source=conf.source,
            target=conf.target,
            prefix=conf.prefix,
            batch_size=conf.batch_size,
            epoch=conf.training.epoch,
            ort=conf.ort,
            from_flax=conf.from_flax,
            num_test_examples=conf.num_test_examples,
            num_validation_examples=conf.num_validation_examples,
            max_train_examples=conf.max_train_examples,
            dataset_processor=conf.dataset_processor,
            learning_rate=conf.training.lr,
            config_file=config_file,
            params=conf.params(),
        )
    print("CONFIG", str(config_file))
    mlflow.log_artifact(str(config_file))


@app.command("optimize")
def optimize(model_path: str, dataset: str, source: str, target: str):
    from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer  # type: ignore
    from optimum.onnxruntime.configuration import (
        AutoCalibrationConfig,
        AutoQuantizationConfig,
    )
    from transformers import AutoTokenizer

    print("\n\n****** CREATING ONNX MODEL ******")
    start = pc()
    model = ORTModelForSeq2SeqLM.from_pretrained(model_path, export=True)
    model = cast(ORTModelForSeq2SeqLM, model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with tempfile.TemporaryDirectory() as output_dir:
        print(f"* Onnx model dir: {output_dir}")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)

        print(f"* Onnx model created in {pc() - start}s")
        from IPython import embed

        model_dir = Path(model_path)
        quantized_dir = model_dir.with_name(model_dir.name + "-quantized")
        encoder_quantizer = ORTQuantizer.from_pretrained(
            output_dir, file_name="encoder_model.onnx"
        )
        decoder_quantizer = ORTQuantizer.from_pretrained(
            output_dir, file_name="decoder_model.onnx"
        )
        decoder_wp_quantizer = ORTQuantizer.from_pretrained(
            output_dir, file_name="decoder_with_past_model.onnx"
        )

        # embed()
        qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)

        def preprocess_fn(ex, tokenizer, source, model):
            return tokenizer(ex[source])

        def preprocess_fn_decoder(ex, tokenizer, source, model):
            res = tokenizer(ex[source], return_tensors="pt", padding=True)
            out = model.encoder(**res)
            res["encoder_attention_mask"] = res.attention_mask
            res["encoder_hidden_states"] = out.last_hidden_state
            return res

        for i, (quantizer, fn) in enumerate(
            [
                (encoder_quantizer, preprocess_fn),
                (decoder_quantizer, preprocess_fn_decoder),
                (decoder_wp_quantizer, preprocess_fn_decoder),
            ]
        ):
            print("QUNATIZE", i + 1)
            calibration_dataset = quantizer.get_calibration_dataset(
                dataset,
                preprocess_function=partial(
                    preprocess_fn, tokenizer=tokenizer, source=source, model=model
                ),
                num_samples=50,
                dataset_split="train",
            )
            calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
            quantizer.fit(
                dataset=calibration_dataset,
                calibration_config=calibration_config,
                operators_to_quantize=qconfig.operators_to_quantize,
            )
            dqconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            )
            quantizer.quantize(
                save_dir=str(quantized_dir),
                # calibration_tensors_range=ranges,
                quantization_config=dqconfig,
            )

            tokenizer.save_pretrained(str(quantized_dir))
            embed()

    # quantized_dir = output_dir.with_name(output_dir.name + "-quantized")
    # encoder_quantizer = ORTQuantizer.from_pretrained(
    #     output_dir, file_name="encoder_model.onnx"
    # )
    # decoder_quantizer = ORTQuantizer.from_pretrained(
    #     output_dir, file_name="decoder_model.onnx"
    # )
    # decoder_wp_quantizer = ORTQuantizer.from_pretrained(
    #     output_dir, file_name="decoder_with_past_model.onnx"
    # )
    # quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]
    # dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    # for q in quantizer:
    #     q.quantize(
    #         save_dir=quantized_dir,
    #         quantization_config=dqconfig,
    #     )
    #
    # prediction_file = Path(model_path) / "eval" / "predictions.csv"
    # if prediction_file.exists():
    #     new_predictions = []
    #     diff = 0
    #     data = pd.read_csv(prediction_file)
    #     total = data.shape[0]
    #     for i, (s, t, p) in enumerate(
    #         zip(data[source], data[target], data[f"{target}_predicted"])
    #     ):
    #         print(f"{i + 1} / {total}")
    #         tokens = tokenizer(s, return_tensors="pt")
    #         input_len = tokens.input_ids.size(1)
    #         if input_len < 10:
    #             max_length = 25
    #         else:
    #             max_length = math.floor(input_len * 1.5) + 5
    #
    #         res = tokenizer.decode(
    #             model.generate(**tokens)[0],
    #             skip_special_tokens=True,
    #             max_length=max_length,
    #         )
    #         res = fix_punctuation(res)
    #         p = fix_punctuation(p)
    #         new_predictions.append(p)
    #         if fix_punctuation(res) != fix_punctuation(p):
    #             diff += 1
    #             print(">> DIFF")
    #             print(fix_punctuation(res))
    #             print(fix_punctuation(p))
    #             print("=====")
    #     print("Total :", total)
    #     print("Diff  :", diff)
    # else:
    #     print("[WARNING] no predictions.csv")


@app.command("eval")
def eval(model: str, data: str, source: str, target: str, prefix: str = ""):
    from .train import eval

    eval(
        model_name=model,
        data=data,
        data_kwargs={},
        max_input_length=256,
        max_target_length=256,
        source=source,
        target=target,
        prefix=prefix,
    )


@app.command()
def repl(model_path: str):
    from transformers import pipeline

    model = pipeline("text-classification", model=model_path)
    print("Loaded model from", model_path)
    while True:
        text = input("> ")
        output = model([text])
        print(">>", output)
        print(">")
