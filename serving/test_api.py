import requests
import json

def test_prediction():
    # API endpoint
    url = "http://localhost:8000/predict"

    # Test data
    payload = {
        "text": [
            "Anggota parlemen Demokrat mengatakan menindak biaya cerukan bank $ 30-35 adalah salah satu cara untuk membantu konsumen yang kesulitan.",
            "Bank Indonesia mempertahankan suku bunga acuan di level 6 persen."
        ]
    }

    # Make POST request
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # Check if request was successful
        response.raise_for_status()

        # Print results
        print("\nTest Results:")
        print("Status Code:", response.status_code)
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")

if __name__ == "__main__":
    test_prediction()
