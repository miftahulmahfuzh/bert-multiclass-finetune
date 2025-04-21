from gradio_client import Client

client = Client("http://10.183.0.2:7861/")
result = client.predict(
		message="Hello!!",
		user_id="101",
		api_name="/predict"
)
print(result)
# for char in result:
#     yield char
