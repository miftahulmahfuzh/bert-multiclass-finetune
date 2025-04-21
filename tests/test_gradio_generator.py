from gradio_client import Client

client = Client("http://10.183.0.2:7861/")
result = client.predict(
    message="berapa harga saham RAJA?",
    user_id="101",
    api_name="/predict"
)
history = ""
query_id = "none"
for chunk in result:
    print(chunk, end="", flush=True)

    # # Yield the current state to update the UI
    # yield history
    # if not query_id and chunk.startswith("{"):
    #     try:
    #         data = json.loads(chunk)
    #         query_id = data.get("query_id")
    #         continue  # Skip adding this to the visible output
    #     except json.JSONDecodeError:
    #         # Not JSON, treat as regular content
    #         history += chunk
    # else:
    #     # Regular content chunk
    #     history += chunk

# print(f"QUERY_ID: {query_id}")
# print(f"FINAL_OUTPUT: {history}")
