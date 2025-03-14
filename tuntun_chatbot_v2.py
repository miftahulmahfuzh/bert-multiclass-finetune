from langchain_core.prompts import PromptTemplate
template = """<|begin_of_text|> <|start_header_id|>system<|end_header_id|>You are a virtual assistant for the Tuntun investment app. Based on the given JSON context or tools output, provide a comprehensive response that accurately answers the query. If the required information is not available, respond with 'I don't know' without referencing any sources or tools used. Ensure your response is complete, clear, and helpful.
<|eot_id|> <|start_header_id|>context<|end_header_id|> {tools_output}
{context} <|eot_id|>
<|start_header_id|>history<|end_header_id|> {chat_history} <|eot_id|>
<|start_header_id|>user<|end_header_id|> {question} <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
prompt = PromptTemplate(
    input_variables=["context", "question", "tools_output", "chat_history"],
    template=template
)
doc_template = "{Answer}"
doc_prompt = PromptTemplate(
    input_variables=["Answer"],
    template=doc_template
)

import time
from datetime import datetime
import gradio as gr
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from core.model import llm_ollama
# from core.prompt import prompt, doc_prompt
from core.rag import vectorstore_none
from tool.tool_caller import process_tools

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="query"  # Changed to match RetrievalQA expected input
)

def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)

def rag_chain(question, history):
    # Load previous conversation from memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Format chat history for the prompt
    formatted_history = ""
    if chat_history:
        formatted_history = "\n".join([
            f"Human: {msg.content}" if i % 2 == 0 else f"Assistant: {msg.content}"
            for i, msg in enumerate(chat_history)
        ])

    # Process tools and create partial prompt
    tools_output = process_tools(question)
    processed_prompt = prompt.partial(
        tools_output=tools_output,
        chat_history=formatted_history
    )

    # Set up QA chain with memory
    qa = RetrievalQA.from_chain_type(
        llm=llm_ollama,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": processed_prompt,
            "document_prompt": doc_prompt,
            "verbose": True
        },
        retriever=vectorstore_none.as_retriever(),
        verbose=True
    )

    print("Timestamp: " + str(datetime.today()))
    partial_message = ""

    # Get response using "query" key instead of "question"
    response = qa.invoke({"query": question}).get("result")

    # Stream response and save to memory
    for char in response:
        partial_message += char
        time.sleep(0.005)
        yield partial_message

    # Save the conversation to memory using "query" as input key
    memory.save_context(
        {"query": question},
        {"output": partial_message}
    )

# Launch Gradio interface
gr.ChatInterface(rag_chain).launch(server_name='0.0.0.0')
