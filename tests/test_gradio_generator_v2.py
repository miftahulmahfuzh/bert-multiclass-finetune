from gradio_client import Client
import json

# Initialize the client
client = Client("http://10.183.0.2:7862/")

history_dir = "/home/devmiftahul/nlp/faq_chatbot/chatbot_13mar_2025/tests/json"
history_json = f"{history_dir}/history_stock_price.json"
# history = json.load(open(history_json))
history_from_fe = open(history_json).read()

# Connect to the API endpoint with streaming enabled
# message="berapa harga saham RAJA?"
message="how about BBCA"
# result_iterator = client.predict(
#     message=message,
#     user_id="102",
#     history=history,
#     api_name="/predict"
# )
# print(result_iterator)

# # Variables to track the response
history = ""
query_id = None

# Process each chunk as it arrives
# for chunk in result_iterator:
for chunk in client.predict(
    message=message,
    user_id="104",
    history=history_from_fe,
    api_name="/predict"
    ):
    # Print the chunk immediately
    print(chunk, end="", flush=True)

    # # Optional: Process the chunk for query_id if needed
    # if not query_id and chunk.startswith("{"):
    #     try:
    #         data = json.loads(chunk)
    #         query_id = data.get("query_id")
    #         # Don't add this to history if it's just metadata
    #         continue
    #     except json.JSONDecodeError:
    #         # Not JSON, treat as regular content
    #         history += chunk
    # else:
    #     # Regular content chunk
    #     history += chunk

# # Print summary information at the end if needed
# print("\n\n--- Summary ---")
# if query_id:
    # print(f"QUERY_ID: {query_id}")
# print(f"FINAL_OUTPUT LENGTH: {len(history)} characters")
