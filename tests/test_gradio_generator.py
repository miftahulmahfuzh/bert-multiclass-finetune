from gradio_client import Client

client = Client("http://10.183.0.2:7861/")
result = client.predict(
    message="Hello!!",
    user_id="101",
    api_name="/predict"
)
# print(result[:5])

# Since result is a generator, iterate through it to get the complete response
# if isinstance(result, type((_ for _ in ()))):  # Check if it's a generator
# First item is the query_id JSON
first_chunk = next(result)
try:
    query_id_data = json.loads(first_chunk)
    query_id = query_id_data.get("query_id")
    print(f"Query ID: {query_id}")

    # Collect the rest of the streamed content
    content = ""
    for chunk in result:
        content += chunk
        # Optionally print each chunk as it arrives
        # print(chunk, end="", flush=True)

    print("\nFull content:")
    print(content)
except json.JSONDecodeError:
    # If the first chunk isn't JSON, it might be content
    content = first_chunk
    for chunk in result:
        content += chunk
    print(content)
# else:
#     # If not a generator, print as is
#     print(result)
