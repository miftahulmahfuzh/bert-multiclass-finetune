import environments
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from typing import List, Dict
from pydantic import BaseModel, Field
import asyncio
import logging
import time
import os
import tiktoken
import json
import redis
import uuid
from datetime import datetime

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define model
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, max_tokens=32768)

# Initialize tokenizer for GPT-4.1 nano (using gpt-4 encoding as a close match)
tokenizer = tiktoken.encoding_for_model("gpt-4")
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Constants
MAX_INPUT_TOKENS = 1_047_576  # GPT-4.1 nano input limit
MAX_OUTPUT_TOKENS = 32_768    # GPT-4.1 nano output limit
ESTIMATED_TOKENS_PER_COMMENT = 50  # Assume 50 tokens per comment
BATCH_SIZE = 2  # Number of posts per batch, adjustable

# Structured output schema
class Comment(BaseModel):
    post_index: int = Field(description="The index of the post (1-based) this comment corresponds to")
    comment: str = Field(description="The generated comment for the post")

class CommentsResponse(BaseModel):
    comments: List[Comment] = Field(description="List of comments, one for each input post")

# Create structured LLM
structured_llm = model.with_structured_output(CommentsResponse)

def add_system_prompt(posts: List[str]) -> List[Dict]:
    """Create a single system prompt and user message for multiple posts."""
    system_str = open(settings.SYSTEM_PROMPT_PATH).read()
    user_content = (
        f"Below is a list of {len(posts)} posts. Generate exactly one comment for each post. "
        "Each comment should be concise and relevant to the post content. "
        "Return the comments as a structured list, where each entry contains the post index (1-based) and the comment text.\n\n"
        "Posts:\n"
    )
    print(f"\nUSER_CONTENT PROMPT: {user_content}\n")
    for i, post in enumerate(posts, 1):
        user_content += f"{i}. {post}\n"

    return [
        {"role": "system", "content": system_str},
        {"role": "user", "content": user_content}
    ]

def count_tokens(messages: List[Dict]) -> int:
    """Count tokens in a list of messages using tiktoken."""
    total_tokens = 0
    for message in messages:
        content = message["content"]
        total_tokens += len(tokenizer.encode(content))
    return total_tokens

def estimate_max_posts(posts: List[str], system_messages: List[Dict]) -> int:
    """Estimate how many posts can be processed within token limits."""
    base_tokens = count_tokens(system_messages)
    remaining_input_tokens = MAX_INPUT_TOKENS - base_tokens
    output_tokens_needed = 0
    posts_count = 0

    for post in posts:
        post_line = f"{posts_count + 1}. {post}\n"
        post_tokens = len(tokenizer.encode(post_line))
        if base_tokens + post_tokens > remaining_input_tokens:
            break
        if output_tokens_needed + ESTIMATED_TOKENS_PER_COMMENT > MAX_OUTPUT_TOKENS:
            break
        base_tokens += post_tokens
        output_tokens_needed += ESTIMATED_TOKENS_PER_COMMENT
        posts_count += 1

    return posts_count

class QueryRequest(BaseModel):
    inputs: List[str]

class QueryResponse(BaseModel):
    number_of_processed_posts: int
    outputs: List[str]

async def process_batch(batch_id: str, posts: List[str], worker_pid: int) -> List[str]:
    """Process a batch of posts and return comments."""
    logger.info(f"Worker PID {worker_pid} processing batch {batch_id} with {len(posts)} posts")
    messages = add_system_prompt(posts)
    try:
        response = await structured_llm.ainvoke(messages)
        # Write response to a unique file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"json/output_{timestamp}_{worker_pid}_{batch_id}.json"
        try:
            os.makedirs("json", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(response.dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Worker PID {worker_pid} failed to write to {output_file}: {str(e)}")

        comments = response.comments
        if len(comments) != len(posts):
            logger.warning(
                f"Worker PID {worker_pid} batch {batch_id}: Mismatch in comment count: "
                f"expected {len(posts)} comments, received {len(comments)}"
            )
        sorted_comments = sorted(comments, key=lambda x: x.post_index)
        if not all(c.post_index == i + 1 for i, c in enumerate(sorted_comments)):
            raise ValueError(f"Worker PID {worker_pid} batch {batch_id}: Invalid or missing post indices")
        return [c.comment for c in sorted_comments]
    except Exception as e:
        logger.error(f"Worker PID {worker_pid} batch {batch_id} failed: {str(e)}")
        raise

@app.post("/mcp_chat", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    process_id = os.getpid()
    logger.info(f"Processing query in worker PID {process_id} with {len(request.inputs)} posts")

    if not request.inputs:
        raise HTTPException(status_code=400, detail="Input list cannot be empty")

    # Split posts into batches
    batches = [request.inputs[i:i + BATCH_SIZE] for i in range(0, len(request.inputs), BATCH_SIZE)]
    queue_name = f"post_batches_{uuid.uuid4()}"
    for batch_idx, batch_posts in enumerate(batches):
        # Estimate max posts for this batch
        base_messages = add_system_prompt([])
        max_posts = estimate_max_posts(batch_posts, base_messages)
        batch_posts = batch_posts[:max_posts]
        if max_posts == 0:
            logger.warning(f"Batch {batch_idx} skipped: Input exceeds token limits")
            continue
        # Push batch to Redis queue
        batch_id = f"batch_{batch_idx}"
        redis_client.rpush(queue_name, json.dumps({"batch_id": batch_id, "posts": batch_posts}))

    # Workers process batches from the queue
    all_comments = []
    while redis_client.llen(queue_name) > 0:
        # Pop a batch from the queue (atomic operation)
        batch_data = redis_client.lpop(queue_name)
        if not batch_data:
            continue
        batch = json.loads(batch_data)
        batch_id = batch["batch_id"]
        batch_posts = batch["posts"]
        # Process the batch
        try:
            comments = await process_batch(batch_id, batch_posts, process_id)
            all_comments.extend(comments)
        except Exception as e:
            logger.error(f"Worker PID {process_id} failed to process batch {batch_id}: {str(e)}")
            continue

    end_time = time.time()
    logger.info(f"Query completed in worker PID {process_id} in {end_time - start_time:.2f} seconds")

    return QueryResponse(
        number_of_processed_posts=len(all_comments),
        outputs=all_comments
    )

@app.on_event("startup")
async def startup_event():
    logger.info(f"Worker PID {os.getpid()} started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Worker PID {os.getpid()} shutting down")
    redis_client.close()
