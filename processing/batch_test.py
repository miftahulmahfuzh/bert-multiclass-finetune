# %% IMPORTS
import gc
import time

import pandas as pd
from langchain.chains.retrieval_qa.base import RetrievalQA

from core.model import llm_ollama
from core.prompt import prompt, doc_prompt
from core.rag import compression_retriever_sim
gc.collect()
processed_prompt = prompt.partial(tools_output="")
qa = RetrievalQA.from_chain_type(llm=llm_ollama,
                                 chain_type="stuff",
                                 chain_type_kwargs={"prompt": processed_prompt,
                                                    "document_prompt": doc_prompt,
                                                    "verbose": False
                                                    },
                                 retriever=compression_retriever_sim,
                                 verbose=False)


def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)


def print_rag(question):
    print(combine_docs(compression_retriever_sim.invoke(question)))


df = pd.read_excel("../dataset/questions-en.xlsx")

# print(print_rag("What options and services does Tuntun offer to newly registered users who haven't opened an RDN account yet?"))

for index, row in df.iterrows():
    print("======================================")
    print("No: " + str(index + 1))
    # print_rag(row.Question)
    start = time.time()
    ans = qa.invoke(row.Question, return_only_outputs='result').get("result")
    end = time.time()
    print("Q: " + row.Question)
    print("A: " + ans)
    print("T: " + str(end - start))
    df.at[index, 'Answer'] = ""
    df.at[index, 'Answer'] = ans
    df.at[index, 'RAG'] = ""
    df.at[index, 'RAG'] = combine_docs(compression_retriever_sim.invoke(row.Question))
df.to_excel("../dataset/questions-en-result.xlsx")
print("===================Finished===================")
