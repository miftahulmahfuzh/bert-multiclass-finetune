import requests
import json

def read_prompt_file(filename):
    """Read content from the prompt file"""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def call_ollama_api(prompt):
    """Make POST request to Ollama API"""
    url = "http://localhost:11435/api/generate"

    payload = {
        "model": "deepseek-r1:32b",
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,      # Set to 1 to disable top-p sampling
        "top_k": 1,      # Set to 1 to always pick the most likely token
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

def write_answer_file(answer, filename):
    """Write LLM response to answer file"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(answer)

def main():
    prompt = read_prompt_file('prompt_v2.txt')
    # fnews = "/home/devmiftahul/nlp/news_tag/news/news_4831022.txt" # a1
    # fnews = "/home/devmiftahul/nlp/news_tag/news/news_4837053.txt" # a2
    # fnews = "/home/devmiftahul/nlp/news_tag/news/news_4831071.txt" # a3
    # news_str = read_prompt_file(fnews)
    news_str = "$TBLA - Taboola Signs memasuki kesepakatan eksklusif baru selama lima tahun dengan Grey Television"
    prompt_content = prompt.replace("<<NEWS>>", news_str)
    print(prompt_content)
    print("============================================")

    # Get LLM response
    llm_response = call_ollama_api(prompt_content)

    if llm_response:
        print(llm_response)
        # Write response to file
        write_answer_file(llm_response, 'answer.txt')
        print("\nResponse successfully written to answer.txt")
    else:
        print("Failed to get response from LLM")

if __name__ == "__main__":
    main()
