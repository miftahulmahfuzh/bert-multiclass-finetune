1. Instruction tuning script:
/home/devmiftahul/nlp/llm_dev/v3/llm_train_v3.py

Instruction tuning output:
/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509

2. Run merge script:
/home/devmiftahul/nlp/llm_dev/olama/merge.py

Merge output:
/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/best-checkpoint/merged_model

3. Run convert to gguf bash script:
(you have to clone llama.cpp first, then inside llama.cpp dir)
git clone https://github.com/ggerganov/llama.cpp
/home/devmiftahul/nlp/llm_dev/olama/llama.cpp/convert.sh

Gguf file:
/home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/gemma-2-2b-it.gguf

4. Create Modelfile, e.g:
FROM /home/devmiftahul/nlp/llm_dev/v3/google/gemma-2-2b-it_20250122_175509/gemma-2-2b-it.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
TEMPLATE """Instruction: Categorize the news text
Input: {{.Prompt}}
Response:"""

5. Run ollama command to convert it to ollama readable:
OLLAMA_HOST=http://localhost:11435 ollama create gemma-2b-classifier -f Modelfile

6. Serve ollama, then load ollama readable:
OLLAMA_HOST=http://localhost:11435 ollama serve
OLLAMA_HOST=http://localhost:11435 ollama run gemma-2b-classifier

7. Run test_ollama.py to test REST api:
/home/devmiftahul/nlp/llm_dev/olama/test_olama.py
python test_olama.py
