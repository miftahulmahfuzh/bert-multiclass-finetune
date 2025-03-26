from langchain_core.prompts import PromptTemplate
from config import settings

template = ""
pv = settings.PROMPT_VERSION.lower()
if pv == "v1":
    from prompt.prompt_v1 import template
elif pv == "v2":
    from prompt.prompt_v2 import template

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
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory

from core.model import llm_ollama
# from core.model_native import hf
from core.rag import vectorstore_none
from tool.tool_caller import process_tools
from db.arango import PyArangoDB

import redis
if settings.REDIS_URL:
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    # to clear redis cache storage, run
    # python -m redis_compose.flush_redis
else:
    redis_client = None

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="query"  # Changed to match RetrievalQA expected input
)

db = PyArangoDB(
    url=settings.LOG_DB_URL,
    username=settings.LOG_DB_USERNAME,
    password=settings.LOG_DB_PASSWORD.get_secret_value(),
    database=settings.LOG_DB_NAME
)

def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)

def get_prev_questions(chat_history):
    prev_questions = []
    for i, msg in enumerate(chat_history):
        if i % 2 == 0:
            prev_questions.append(msg.content)
    return prev_questions

def rag_chain(question, stream=True):
    # Load previous conversation from memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Format chat history for the prompt
    formatted_history = ""
    if chat_history:
        formatted_history = "\n".join([
            f"Human: {msg.content}" if i % 2 == 0 else f"Assistant: {msg.content}"
            for i, msg in enumerate(chat_history)
        ])
    prev_questions = get_prev_questions(chat_history)


    print("Timestamp: " + str(datetime.today()))
    partial_message = ""
    qa = None
    use_llm = True

    # Create cache key using last 2 previous questions + current question
    cache_key = " | ".join(prev_questions[-2:] + [question])
    cache_key = f"ai_chatbot_conv:{cache_key}"

    # Manual Redis caching implementation
    if redis_client:
        print(f"Search cache_key: {cache_key}")
        # Check if answer exists in cache
        cached_response = redis_client.get(cache_key)
        if cached_response:
            # If found in cache, use cached response
            print(f"Use answer from Redis:\n{cached_response}")
            response = cached_response
            use_llm = False

    if use_llm:
        # Process tools and create partial prompt
        tools_output, selected_tools, selected_tools_timestamp = process_tools(question, prev_questions)
        processed_prompt = prompt.partial(
            tools_output=tools_output,
            chat_history=formatted_history
        )

        # Set up QA chain with memory
        qa = RetrievalQA.from_chain_type(
            llm=llm_ollama,
            # llm=hf,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": processed_prompt,
                "document_prompt": doc_prompt,
                "verbose": True
            },
            retriever=vectorstore_none.as_retriever(),
            verbose=True
        )

        response = qa.invoke({"query": question}).get("result")
        if redis_client:
            if all(tool not in settings.SKIP_TOOLS for tool in selected_tools):
                print(f"Selected tools by LLM: {selected_tools}")
                print("Insert query and answer to cache")
                print(f"New cache_key: {cache_key}")
                redis_client.setex(cache_key, 86400, response)

    partial_message = ""
    if stream:
        for char in response:
            partial_message += char
            time.sleep(0.005)
            yield partial_message
    else:
        yield response  # Return full response immediately

    # Save the conversation to memory using "query" as input key
    memory.save_context(
        {"query": question},
        {"output": partial_message if stream else response}
    )

    # doc = db.create_chat_log(
    #     user_id=101,
    #     query_id=1001,
    #     channel=1,
    #     user_query="What is the weather?",
    #     final_input="User asked: What is the weather?",
    #     final_output="The weather is sunny!",
    #     reaction=1,
    #     selected_tools=["weather_api", "stock_price"],
    #     user_query_timestamp=now,
    #     final_input_timestamp=now,
    #     final_output_timestamp=now,
    #     reaction_timestamp=now,
    #     selected_tools_timestamp=now
    # )

if __name__=="__main__":
    # Launch Gradio interface
    import gradio as gr
    gr.ChatInterface(rag_chain).launch(server_name='0.0.0.0')
