# Comment Generation Service

This project provides a backend service for generating concise, conversational comments in Indonesian based on investment or stock-related posts. It is designed to mimic the natural, casual language used by Indonesian investors on social media.

---

## Features

- Accepts multiple stock-related posts as input.
- Generates one relevant comment per post in Indonesian.
- Uses a large language model (LLM) with structured output for reliable responses.
- Handles token limits by estimating how many posts can be processed in one request.
- Provides a FastAPI-based HTTP API for easy integration.

---

## Project Structure

- `main_app_v2.py`: Main FastAPI application that handles requests, prepares prompts, calls the LLM, and returns structured comments.
- `config.py`: Configuration settings using Pydantic, including API keys, model settings, and environment variables.
- `api_openai/prompt/system_comment.txt`: System prompt template guiding the LLM to generate comments in the desired style.
- `.env`: Environment variables including API keys and other secrets.

---

## How It Works

1. The API receives a list of stock-related posts.
2. It constructs a system prompt and a user message listing all posts.
3. It estimates how many posts can be processed within the model's token limits.
4. It sends the prompt to the LLM, requesting one comment per post.
5. The LLM returns a structured response with comments indexed by post.
6. The API returns the comments in order.

---

## Deployment

### Prerequisites

- Python 3.9 or higher
- Access to OpenAI API or compatible LLM API
- Required Python packages (see `requirements.txt`)

### Setup

1. Clone the repository.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` file:
   - `OPENAI_API_KEY`: Your OpenAI API key or equivalent.
   - Other keys as needed.

5. Adjust settings in `config.py` if necessary.

### Running the Service

Run the FastAPI app using Uvicorn:
```bash
uvicorn main_app_v2:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

---

## API Usage

### Endpoint: `/mcp_chat`

- Method: POST
- Request Body:
  ```json
  {
    "inputs": ["post1", "post2", ...]
  }
  ```
- Response:
  ```json
  {
    "number_of_processed_posts": n,
    "outputs": ["comment1", "comment2", ...]
  }
  ```

The service processes as many posts as possible within token limits and returns one comment per post.

---

## Extending and Improving

- Modify the system prompt in `api_openai/prompt/system_comment.txt` to change the comment style or language.
- Adjust token limits and model parameters in `main_app_v2.py`.
- Integrate with other LLM providers by updating the model initialization.
- Add caching or logging for better performance and traceability.

---

## Notes

- The service uses GPT-4.1 nano model with a large token limit.
- Token counting is done with `tiktoken` to ensure prompt size compliance.
- The system prompt guides the LLM to produce comments in a casual, conversational Indonesian style.

---

## License

Specify your license here.

---

## Contact

For questions or contributions, please contact the maintainer.

