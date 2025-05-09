import environments
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel as PydanticBaseModel, Field
from typing import List, Dict
from pydantic import BaseModel
import asyncio
import logging
import time
import os
import tiktoken

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define model
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, max_tokens=32768)

# Initialize tokenizer for GPT-4.1 nano (using gpt-4 encoding as a close match)
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Constants
MAX_INPUT_TOKENS = 1_047_576  # GPT-4.1 nano input limit
MAX_OUTPUT_TOKENS = 32_768    # GPT-4.1 nano output limit
ESTIMATED_TOKENS_PER_COMMENT = 50  # Assume 50 tokens per comment

# Structured output schema
class Comment(PydanticBaseModel):
    post_index: int = Field(description="The index of the post (1-based) this comment corresponds to")
    comment: str = Field(description="The generated comment for the post")

class CommentsResponse(PydanticBaseModel):
    comments: List[Comment] = Field(description="List of comments, one for each input post")

# Create structured LLM
structured_llm = model.with_structured_output(CommentsResponse)

def add_system_prompt(posts: List[str]) -> List[Dict]:
    """Create a single system prompt and user message for multiple posts."""
    system_str = open(settings.SYSTEM_PROMPT_PATH).read()
    # Construct a user message that lists all posts with instructions
    user_content = (
        "Below is a list of posts. Generate exactly one comment for each post. "
        "Each comment should be concise and relevant to the post content. "
        "Return the comments as a structured list, where each entry contains the post index (1-based) and the comment text.\n\n"
        "Posts:\n"
    )
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
        # Count tokens for the post as it appears in the user message
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

@app.post("/mcp_chat", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    process_id = os.getpid()
    logger.info(f"Processing query in worker PID {process_id} with {len(request.inputs)} posts")

    if not request.inputs:
        raise HTTPException(status_code=400, detail="Input list cannot be empty")

    # Create base system messages (without posts)
    base_messages = add_system_prompt([])

    # Estimate how many posts can be processed
    max_posts = estimate_max_posts(request.inputs, base_messages)
    if max_posts == 0:
        raise HTTPException(status_code=400, detail="Input exceeds token limits")

    # Take only the processable posts
    posts_to_process = request.inputs[:max_posts]
    logger.info(f"Processing {max_posts} out of {len(request.inputs)} posts")

    # Create messages with the selected posts
    messages = add_system_prompt(posts_to_process)

    try:
        # Call the structured LLM once
        response = await structured_llm.ainvoke(messages)
        comments = response.comments
        if len(comments) != len(posts_to_process):
            raise ValueError("Mismatch between number of comments and posts")
        # Ensure comments are in the correct order and extract comment text
        sorted_comments = sorted(comments, key=lambda x: x.post_index)
        if not all(c.post_index == i + 1 for i, c in enumerate(sorted_comments)):
            raise ValueError("Invalid or missing post indices in response")
        comment_texts = [c.comment for c in sorted_comments]
    except Exception as e:
        logger.error(f"Error in worker PID {process_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model invocation failed: {str(e)}")

    end_time = time.time()
    logger.info(f"Query completed in worker PID {process_id} in {end_time - start_time:.2f} seconds")

    return QueryResponse(
        number_of_processed_posts=len(posts_to_process),
        outputs=comment_texts
    )

@app.on_event("startup")
async def startup_event():
    logger.info(f"Worker PID {os.getpid()} started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Worker PID {os.getpid()} shutting down")
