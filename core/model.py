from chromadb.utils.embedding_functions import create_langchain_embedding
# from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config import settings, LLMType

os.environ[
    "OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)
if settings.LLM_TYPE == LLMType.OLLAMA:
    llm = ChatOllama(
        model="llama3.1:70b-instruct-q5_0",
        temperature=0)

EMBEDDING_MODEL_NAME = "dunzhang/stella_en_400M_v5"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                        model_kwargs={'device': 'cuda', 'trust_remote_code': True})
base_embedding_function = create_langchain_embedding(embedding_model)
