import requests

url = "http://localhost:3000/predict"
data = {"input_data": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}
response = requests.post(url, json=data)

print(response.json())
