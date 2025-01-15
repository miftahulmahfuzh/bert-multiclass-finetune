import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from pathlib import Path

def convert_to_onnx(
    input_dir: Path = "",
    output_dir: Path = "",
):
    """
    Convert a saved SafeTensors model to ONNX format using Optimum

    Args:
        input_dir: Directory containing the SafeTensors model and tokenizer files
        output_dir: Directory where the ONNX model will be saved
    """
    print(f"Loading model and tokenizer from {input_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        input_dir,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        input_dir,
        local_files_only=True
    )

    # Convert to ONNX using Optimum
    print("Converting model to ONNX format...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_id=input_dir,
        export=True
    )

    # Save the ONNX model and tokenizer
    print(f"Saving ONNX model and tokenizer to {output_dir}")
    ort_model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # Verify the saved files
    saved_files = os.listdir(output_dir)
    print("\nSaved files:")
    for file in saved_files:
        print(f"- {file}")

    print("\nConversion completed successfully!")
    print(f"\nYou can now load this model using the transformers pipeline:")
    print("from transformers import pipeline")
    print("from optimum.onnxruntime import ORTModelForSequenceClassification")
    print(f'classifier = pipeline("text-classification", model="{output_dir}", device="cpu")')

if __name__ == "__main__":
    d = "/home/devmiftahul/nlp/bert_dev/indobenchmark/indobert-base-p2_20250114_171614"
    convert_to_onnx(
        input_dir=d,
        output_dir=d
    )
