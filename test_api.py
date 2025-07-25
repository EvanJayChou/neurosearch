import requests

API_URL = "http://localhost:8000/chat"

# Example neuroscience dataset prompt
messages = [
    {"role": "user", "content": "Find open-access EEG and MRI datasets with detailed metadata for cognitive neuroscience research."}
]

response = requests.post(
    API_URL,
    json={"messages": messages}
)

if response.ok:
    print("AI Response:")
    print(response.json()["response"])
else:
    print(f"Error: {response.status_code}")
    print(response.text) 