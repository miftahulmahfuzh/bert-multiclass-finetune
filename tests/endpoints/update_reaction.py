import requests
from config import settings

url = "http://35.247.154.29:8000/update_reaction?query_id=34&reaction=dislike"

payload={}
headers = {
  'x-api-key': settings.API_KEY.get_secret_value()
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
