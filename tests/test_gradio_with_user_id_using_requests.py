import requests

url = "http://10.183.0.2:7861/predict"
payload = {
    "message": "Hello!!",
    "user_id": "101"
}

response = requests.post(url, json=payload)
result = response.json()
print(result)
