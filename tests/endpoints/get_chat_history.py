import requests
from config import settings

url = "http://35.247.154.29:8000/get_chat_history?user_id=101"

payload={}
headers = {
  'x-api-key': settings.API_KEY.get_secret_value()
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
