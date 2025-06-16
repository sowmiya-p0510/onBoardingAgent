import requests
import json

def test_onboard_endpoint():
    url = "http://localhost:8000/onboard"
    
    payload = {
        "name": "Test User",
        "role": "Software Developer",
        "start_date": "June 20, 2025"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        print("Status Code:", response.status_code)
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing onboarding endpoint...")
    test_onboard_endpoint()
