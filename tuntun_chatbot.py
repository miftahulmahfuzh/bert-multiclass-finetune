# %% IMPORTS
from datetime import datetime

import gradio as grq
from langchain.chains.retrieval_qa.base import RetrievalQA

from core.model import llm_ollama
from core.prompt import prompt, doc_prompt
from core.rag import vectorstore_none
from tool.tool_caller import process_tools


def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)


def rag_chain(question, history):
    processed_prompt = prompt.partial(tools_output=process_tools(question))
    qa = RetrievalQA.from_chain_type(llm=llm_ollama,
                                     chain_type="stuff",
                                     chain_type_kwargs={"prompt": processed_prompt,
                                                        "document_prompt": doc_prompt,
                                                        "verbose": True
                                                        },
                                     retriever=vectorstore_none.as_retriever(),
                                     verbose=True)
    print("Timestamp: " + str(datetime.today()))
    return qa.invoke(question,
                     return_only_outputs='result').get(
        "result")
    # return json.loads(faq_selector(question))["data"]

    #     # partial_message = ""
    #     # for response in qa.invoke(question, return_only_outputs='result').get("result"):
    #     #     # for response in res:
    #     #     partial_message += response
    #     #     # time.sleep(0.001)
    #     #     yield partial_message
    #
    #


# %% Gradio
gr.ChatInterface(rag_chain).launch(server_name='0.0.0.0')
