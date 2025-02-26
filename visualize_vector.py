import numpy as np
import pandas as pd
from renumics import spotlight
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb import Settings, EmbeddingFunction, Documents, Embeddings

# %% Embedding Function
# EMBEDDING_MODEL_NAME = "Salesforce/SFR-Embedding-2_R"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"


class NVEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents
        sentences = input
        model = SentenceTransformer(EMBEDDING_MODEL_NAME,
                                    device='cuda', trust_remote_code=True)
        embeddings = model.encode(sentences)
        # Convert embeddings to a list of lists
        embeddings_as_list = [embedding.tolist() for embedding in embeddings]
        return embeddings_as_list


nvembeddings = NVEmbeddingFunction()
persist_directory = "./chromadb"
# loader = TextLoader(dataset_directory,
#                     encoding='UTF-8')

client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(allow_reset=True))

collection = client.get_or_create_collection("faq_data", embedding_function=nvembeddings)
# collection = client.get_or_create_collection("test_data", embedding_function=embeddings)
response = collection.get(include=["metadatas", "documents", "embeddings"])
question = "Sis, I'm not selling this, please cancel my order"
df = pd.DataFrame({
    "id": response["ids"],
    "answer": [metadata.get("Answer") for metadata in response["metadatas"]],
    "document": response["documents"],
    "category": [metadata.get("Category") for metadata in response["metadatas"]],
    "embedding": response["embeddings"],
})
# df["contains_answer"] = df["document"].apply(lambda x: question in x)
# %% execute questions
question_embedding = nvembeddings([question])
df["dist"] = df.apply(
    lambda row: np.linalg.norm(
        np.array(row["embedding"]) - question_embedding
    ),
    axis=1,
)
spotlight.show(df)
