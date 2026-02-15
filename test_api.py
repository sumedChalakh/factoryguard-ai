import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "machine_id": 1,
    "temperature": 60,
    "vibration": 29,
    "pressure": 102
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
