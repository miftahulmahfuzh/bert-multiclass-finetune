from gradio_client import Client

client = Client("http://10.183.0.2:7860/")
result = client.predict(
		message="<user_id>104</user_id><query>apa berita terbaru ADHI?</query>",
		api_name="/chat"
)
print(result)
