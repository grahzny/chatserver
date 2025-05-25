import requests
import json

# Test script to verify tools are being passed correctly

url = "http://localhost:8001/v1/chat/completions"

# Test payload with tools
payload = {
    "model": "gemma3-27b",  # Update this to match your model name
    "messages": [
        {"role": "user", "content": "What's the current time?"}
    ],
    "tools": [
        {
            "name": "get_current_datetime",
            "description": "Returns the current date and time in ISO 8601 format",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "calculator",
            "description": "Calculate mathematical expressions safely",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ],
    "tool_choice": "auto",
    "max_tokens": 512,
    "temperature": 0.7
}

print("Sending request to:", url)
print("Payload:", json.dumps(payload, indent=2))

try:
    response = requests.post(url, json=payload)
    print("\nResponse status:", response.status_code)
    print("Response body:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", str(e))
    if hasattr(e, 'response') and e.response:
        print("Response text:", e.response.text)

# Also test the debug endpoint
debug_url = "http://localhost:8001/debug/tools"
print("\n\nTesting debug endpoint...")
debug_response = requests.post(debug_url, json=payload)
print("Debug response:", json.dumps(debug_response.json(), indent=2))