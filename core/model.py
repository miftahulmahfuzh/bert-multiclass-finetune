from chromadb.utils.embedding_functions import create_langchain_embedding
# from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config import settings

os.environ[
    "OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

# llm_model = ChatOllama(model='llama3.1:70b')
# llm_model = ChatOllama(model='llama3.1')
# llm_model = ChatOllama(model='llama3.1:8b-instruct-fp16')

# llm_model = ChatOpenAI(
#     api_key="ollama",
#     # model="llama3.1",
#     model="llama3.1:8b-instruct-fp16",
#     base_url="http://localhost:11434/v1"
# )
#
# llm_model_tool = ChatOpenAI(
#     api_key="ollama",
#     # model="llama3.1",
#     model="llama3.1:8b-instruct-fp16",
#     base_url="http://localhost:11434/v1"
# )

# llm_ollama = ChatDeepSeek(
#     model="deepseek-chat",
#     # model="deepseek-reasoner",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key="sk-ebf3dce87b254fbe8f40bced4141777c"
# )

## Default
llm_ollama = ChatOllama(
    model="deepseek-r1:32b",
    # model="llama3.1:70b",
    # model="llama3.1:70b-instruct-q5_0",
    # model="llama3.2",
    # model="llama3:70b",
    temperature=0)

llm_openai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)

# EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
# EMBEDDING_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
EMBEDDING_MODEL_NAME = "dunzhang/stella_en_400M_v5"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                        model_kwargs={'device': 'cuda', 'trust_remote_code': True})
base_embedding_function = create_langchain_embedding(embedding_model)
