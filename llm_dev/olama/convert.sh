MERGED_MODEL_PATH=/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/best-checkpoint/merged_model
GGUF_MODEL_PATH=/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/gemma-2-2b-it.gguf
python3 convert_hf_to_gguf.py \
    "$MERGED_MODEL_PATH" \
    --outfile "$GGUF_MODEL_PATH" \
    --outtype f16  # or any supported quantization
