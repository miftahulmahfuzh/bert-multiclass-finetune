import chromadb
from chromadb import Settings
from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_chroma import Chroma
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from core.model import base_embedding_function

# persist_directory = "/home/devhermawan/project/chatbot/chromadb"
persist_directory = "/mnt/c/Users/mahfu/Downloads/tuntun/tuntun_ubuntu/hermawan/chatbot/chromadb"

client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(allow_reset=True))

vectorstore_txt = Chroma(client=client, collection_name="faq_txt", embedding_function=base_embedding_function)
vectorstore_none = Chroma(client=client, collection_name="none", embedding_function=base_embedding_function)
vectorstore = Chroma(client=client, collection_name="faq_data", embedding_function=base_embedding_function)

# %% 4. RAG Setup
redundant_filter = EmbeddingsRedundantFilter(embeddings=base_embedding_function)
pipeline = DocumentCompressorPipeline(transformers=[redundant_filter])
retriever_sim = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
retriever_txt = vectorstore_txt.as_retriever(search_type="similarity", search_kwargs={'k': 5})
retriever_none = vectorstore_none.as_retriever(search_type="similarity", search_kwargs={'k': 1})

# retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5})
retriever_merged = MergerRetriever(retrievers=[retriever_sim, retriever_txt])
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=retriever_merged
)

compression_retriever_sim = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=retriever_sim
)
