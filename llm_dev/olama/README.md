# Gemma-2B Classifier: Instruction Tuning, Merging, and Ollama Deployment

This guide walks you through the process of instruction tuning, merging, converting, and deploying a Huggingface Finetuned LLM model for deployment using Ollama.

---

## 1. Instruction Tuning

Run the instruction tuning script:

```bash
python /home/devmiftahul/nlp/llm_dev/v3/llm_train_v3.py
```

- **Output directory:**
  `/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509`

---

## 2. Merge the Model

Run the merge script to combine model checkpoints:

```bash
python /home/devmiftahul/nlp/llm_dev/olama/merge.py
```

- **Merged model output:**
  `/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/best-checkpoint/merged_model`

---

## 3. Convert to GGUF Format

1. **Clone llama.cpp** (if not already done):

    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    ```

2. **Run the conversion script** inside the `llama.cpp` directory:

    ```bash
    /home/devmiftahul/nlp/llm_dev/olama/llama.cpp/convert.sh
    ```

- **GGUF file output:**
  `/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/gemma-2-2b-it.gguf`

---

## 4. Create a Modelfile

Create a `Modelfile` with the following content (adjust paths and parameters as needed):

```Dockerfile
FROM /home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/gemma-2-2b-it.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
TEMPLATE """Instruction: Categorize the news text
Input: {{.Prompt}}
Response:"""
```

---

## 5. Convert to Ollama Format

Use the `ollama` command to create an Ollama-readable model:

```bash
OLLAMA_HOST=http://localhost:11435 ollama create gemma-2b-classifier -f Modelfile
```

---

## 6. Serve and Run the Model with Ollama

Start the Ollama server and run your model:

```bash
OLLAMA_HOST=http://localhost:11435 ollama serve
OLLAMA_HOST=http://localhost:11435 ollama run gemma-2b-classifier
```

---

## 7. Test the Ollama REST API

Run the provided test script to verify the REST API:

```bash
python /home/devmiftahul/nlp/llm_dev/olama/test_ollama.py
```

---

## File Reference

- **Instruction tuning script:**
  `/home/devmiftahul/nlp/llm_dev/v3/llm_train_v3.py`
- **Merge script:**
  `/home/devmiftahul/nlp/llm_dev/olama/merge.py`
- **Conversion script:**
  `/home/devmiftahul/nlp/llm_dev/olama/llama.cpp/convert.sh`
- **Test script:**
  `/home/devmiftahul/nlp/llm_dev/olama/test_ollama.py`

---

## Notes

- Ensure all paths are correct for your environment.
- You must have [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp) installed.
- Adjust parameters in the `Modelfile` as needed for your use case.

---
