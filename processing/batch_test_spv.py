# %% IMPORTS
import time

import pandas as pd
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from core.model import llm_ollama
from core.rag import retriever_none

experts = "'investment_expert', 'stock_expert', 'mutual_fund_expert', 'tuntun_product_expert', 'customer_service'"
template = "System: You are a supervisor responsible for selecting the next worker from the following list: " + experts + ". Based on the user's request, choose the most suitable worker to act next. Respond only with the worker name. When all task is finished, respond with 'finish'.\nHuman: {question}.\nSystem: Who should act next? Or should we finish? Select one of: ['finish', " + experts + "]{context}"
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

llm_func = OllamaFunctions(model="llama3.1:70b-instruct-q5_0", format="json")

qa = RetrievalQA.from_chain_type(llm=llm_ollama,
                                 chain_type="stuff",
                                 chain_type_kwargs={
                                     "prompt": PromptTemplate(
                                         template=template,
                                         input_variables=["context", "question"],
                                     ),
                                 },
                                 retriever=retriever_none,
                                 verbose=False)


def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)


def print_rag(question):
    print(combine_docs(retriever_none.invoke(question)))


df = pd.read_excel("../dataset/agents.xlsx")

# print(print_rag("What options and services does Tuntun offer to newly registered users who haven't opened an RDN account yet?"))

for index, row in df.iterrows():
    print("======================================")
    print("No: " + str(index + 1))
    # print_rag(row.Question)
    start = time.time()
    ans = qa.invoke(row.Questions, return_only_outputs='result').get("result")
    end = time.time()
    print("Q: " + row.Questions)
    print("A: " + ans)
    print("T: " + str(end - start))
    df.at[index, 'Answer'] = ""
    df.at[index, 'Answer'] = ans
    df.at[index, 'RAG'] = ""
    df.at[index, 'RAG'] = combine_docs(retriever_none.invoke(row.Questions))
df.to_excel("../dataset/agents-result.xlsx")
print("===================Finished===================")
