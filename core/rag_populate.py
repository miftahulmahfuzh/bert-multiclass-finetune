import pandas as pd

from core.model import base_embedding_function
from core.rag import client

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# %% Initialize Vector Database

client.reset()

# # %% Initialize txt Data
# dataset_directory_txt = "../dataset/faq.txt"
# loader = TextLoader(dataset_directory_txt,
#                     encoding='UTF-8')
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
# splits = text_splitter.split_documents(docs)
# answers_txt = []
# documents_txt = []
#
# for x in splits:
#     answers_txt.append({"Answer": x.page_content})
#     documents_txt.append(x.page_content)
# collection_txt = client.get_or_create_collection("faq_txt", embedding_function=base_embedding_function)
# collection_txt.add(documents=documents_txt, metadatas=answers_txt,
#                    ids=[str(index) for index, _ in enumerate(documents_txt)])

# %% Initialize Data
dataset_directory = "../dataset/faq-new-format.xlsx"
passage_data = pd.read_excel(dataset_directory)
documents = passage_data['Question'].tolist()
answers = []
categories = []

for idx in range(len(passage_data)):
    answers.append({"Answer": passage_data['Answer'][idx], "Category": passage_data['Category'][idx]})

collection = client.get_or_create_collection("faq_data", embedding_function=base_embedding_function)
collection.add(documents=documents, metadatas=answers, ids=[str(index) for index, _ in enumerate(documents)])
