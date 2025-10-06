import requests
import json

# Create sample input (60 timesteps x 33 features)
payload = {
    "sequence": [[0.01 * i] * 33 for i in range(60)]
}

# Make prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json=payload
)

print(json.dumps(response.json(), indent=2))