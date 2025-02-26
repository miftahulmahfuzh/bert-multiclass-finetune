import os

from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "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"

llm_groq = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
