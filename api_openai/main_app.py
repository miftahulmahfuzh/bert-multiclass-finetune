import environments
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from typing import List, Dict
from pydantic import BaseModel
import asyncio
import logging
import time
import os

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define model
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, max_tokens=1000)

def add_system_prompt(messages: List[Dict]):
    system_str = open(settings.SYSTEM_PROMPT_PATH).read()
    system_head = [{"role": "system", "content": system_str}]
    new_messages = system_head + messages
    return new_messages

class QueryRequest(BaseModel):
    query: str

@app.get("/chat")
async def process_query(query: str):
    start_time = time.time()
    process_id = os.getpid()
    logger.info(f"Processing query in worker PID {process_id}: {query}")

    # Create message list with the query
    messages = [{"role": "user", "content": query}]
    messages = add_system_prompt(messages)

    try:
        response = await model.ainvoke(messages)
    except Exception as e:
        logger.error(f"Error in worker PID {process_id}: {str(e)}")
        raise

    end_time = time.time()
    logger.info(f"Query completed in worker PID {process_id} in {end_time - start_time:.2f} seconds")

    # Return the response
    return {"response": response.content}

@app.on_event("startup")
async def startup_event():
    logger.info(f"Worker PID {os.getpid()} started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Worker PID {os.getpid()} shutting down")
