from langchain_core.prompts import PromptTemplate
from config import settings
import json

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

from core.model import llm_natural_answer_generation
from core.rag import vectorstore_none
from tool.tool_caller import process_tools
from utils import remove_tag_content

from db.arango import PyArangoDB
from db.utils import get_current_timestamp

import redis
if settings.REDIS_URL:
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    # to clear redis cache storage, run
    # python -m redis_compose.flush_redis
else:
    redis_client = None

db = PyArangoDB(
    url=settings.LOG_DB_URL,
    username=settings.LOG_DB_USERNAME,
    password=settings.LOG_DB_PASSWORD.get_secret_value(),
    database=settings.LOG_DB_NAME
)

def get_formatted_history_from_db(user_id, n):
    formatted_history = ""
    prev_questions = []
    if db.connect():
        result = db.get_chat_history(user_id, n)
        history = result["items"]
        formatted_history = "\n".join(f"Human: {x['user_query']}\nAssistant: {x['final_output']}" for x in history)
        prev_questions = [x["user_query"] for x in history]
    return formatted_history, prev_questions

# def rag_chain(question, user_id="101", stream=True):
def rag_chain(question: str, user_id:str = "101"):
    print(f"USER_ID: {user_id}")
    user_query_timestamp = get_current_timestamp()

    n = settings.HISTORY_ITEMS
    formatted_history, prev_questions = get_formatted_history_from_db(user_id, n)

    print("Timestamp: " + str(datetime.today()))
    partial_message = ""
    qa = None
    use_llm = True
    response = ""

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
            response = remove_tag_content(response)
            use_llm = False

    tools_output = ""
    selected_tools = []
    now = get_current_timestamp()
    selected_tools_timestamp = now
    processed_prompt = prompt.partial(
        tools_output=tools_output,
        chat_history=formatted_history
    )
    final_processed_prompt = processed_prompt.format(
        context="",  # Add context if needed
        question=question
    )
    final_input_timestamp = now
    if use_llm:
        # Process tools and create partial prompt
        tools_output, selected_tools, selected_tools_timestamp = process_tools(question, prev_questions)

        processed_prompt = prompt.partial(
            tools_output=tools_output,
            chat_history=formatted_history
        )
        final_processed_prompt = processed_prompt.format(
            context="",  # Add context if needed
            question=question
        )
        final_input_timestamp = get_current_timestamp()

        # Set up QA chain with memory
        qa = RetrievalQA.from_chain_type(
            llm=llm_natural_answer_generation,
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
        # response = remove_tag_content(response)
        if redis_client:
            if all(tool not in settings.TIMEBOUND_TOOLS for tool in selected_tools):
                print(f"Selected tools by LLM: {selected_tools}")
                print("Insert query and answer to cache")
                print(f"New cache_key: {cache_key}")
                redis_client.setex(cache_key, 86400, response)
    stream = False
    # V1 - support stream message
    # partial_message = ""
    # if stream:
    #     for char in response:
    #         partial_message += char
    #         time.sleep(0.005)
    #         yield partial_message
    # else:
    #     yield response  # Return full response immediately

    final_output_timestamp = get_current_timestamp()

    if db.connect():
        now = get_current_timestamp()
        query_id = db.create_chat_log(
            user_id=user_id,
            user_query=question,
            final_input=final_processed_prompt,
            final_output=partial_message if stream else response,
            reaction="none",
            selected_tools=selected_tools,
            user_query_timestamp=user_query_timestamp,
            final_input_timestamp=final_input_timestamp,
            final_output_timestamp=final_output_timestamp,
            reaction_timestamp=now,
            selected_tools_timestamp=selected_tools_timestamp
        )

    # V2 - this is used in gradio_ui.py
    # result = {"final_output": response, "query_id": query_id}
    # result_str = json.dumps(result, indent=3)
    # return result_str

    # V3 - Return the result based on streaming mode. used in gradio_ui_v2.py
    # print(f"STREAM: {stream}")
    # if stream:
    #     print(f"STREAM IS SET TO TRUE")
    #     # First yield the query_id as a special message
    #     first_chunk = json.dumps({"query_id": query_id})
    #     yield first_chunk

    #     # Then yield the content character by character
    #     for char in response:
    #         time.sleep(0.005)
    #         yield char
    # else:
    #     print(f"STREAM IS SET TO FALSE")
    #     # Non-streaming mode - return everything as a single JSON object
    #     result = {"final_output": response, "query_id": query_id}
    #     return json.dumps(result, indent=3)
    result = {"final_output": response, "query_id": query_id}
    return result

def rag_chain_dict(question: str, user_id: str= "101"):
    return rag_chain(question, user_id)

def rag_chain_stream(question: str, user_id: str= "101"):
    result = rag_chain(question, user_id)
    query_id = result["query_id"]
    response = result["final_output"]

    # First yield the query_id as a special message
    first_chunk = json.dumps({"query_id": query_id})
    yield first_chunk

    # Then yield the content character by character
    for char in response:
        time.sleep(0.005)
        yield char
