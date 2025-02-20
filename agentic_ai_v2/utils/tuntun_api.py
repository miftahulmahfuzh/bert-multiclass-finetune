import requests
import json

def get_basic_trading_data(secCode, startDate, endDate):
    url = "http://10.192.1.245:8080/orderbook/basic-trading-data"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "secCode": secCode,
        "startDate": startDate,
        "endDate": endDate
    }

    # Sending POST request
    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Parse the JSON response
            data = response.json()
            if data.get("errorCode") == "FH000000":
                return data["data"]  # Return the data part if the request was successful
            else:
                print(f"Error: {data.get('message')}")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON response")
    else:
        print(f"Error: Request failed with status code {response.status_code}")

if __name__=="__main__":
    # Example usage
    secCode = "BAYU"
    startDate = "2024-03-08"
    endDate = "2024-03-13"
    trading_data = get_basic_trading_data(secCode, startDate, endDate)
    print(trading_data)

