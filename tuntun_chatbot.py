# %% IMPORTS
import time
from datetime import datetime

import gradio as gr
from langchain.chains.retrieval_qa.base import RetrievalQA

from core.model import llm_ollama
from core.prompt import prompt, doc_prompt
from core.rag import vectorstore_none
from tool.tool_caller import process_tools


def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)

prev_questions = []

def rag_chain(question, history):
    global prev_questions
    prev_questions_str = "\n".join(prev_questions)
    processed_prompt = prompt.partial(tools_output=process_tools(question), prev_questions=prev_questions_str)
    qa = RetrievalQA.from_chain_type(llm=llm_ollama,
                                     chain_type="stuff",
                                     chain_type_kwargs={"prompt": processed_prompt,
                                                        "document_prompt": doc_prompt,
                                                        "verbose": True
                                                        },
                                     retriever=vectorstore_none.as_retriever(),
                                     verbose=True)
    print("Timestamp: " + str(datetime.today()))
    # return qa.invoke(question,
    #                  return_only_outputs='result').get(
    #     "result")
    # return json.loads(faq_selector(question))["data"]

    partial_message = ""
    for response in qa.invoke(question, return_only_outputs='result').get("result"):
        # for response in res:
        partial_message += response
        time.sleep(0.005)
        yield partial_message

    prev_questions.append(question)
    #
    #


# %% Gradio
gr.ChatInterface(rag_chain).launch(server_name='0.0.0.0')
