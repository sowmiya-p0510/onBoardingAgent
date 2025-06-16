import requests
import json

def test_benefits_endpoint():
    url = "http://localhost:8000/benefits"
    
    # Test case 1: Request without a question
    payload1 = {
        "role": "Software Developer"
    }
    
    # Test case 2: Request with a question
    payload2 = {
        "role": "Software Developer",
        "question": "What health insurance options do I have?"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing benefits endpoint without question...")
        response1 = requests.post(url, json=payload1, headers=headers)
        response1.raise_for_status()
        
        print("Status Code:", response1.status_code)
        print("Response:")
        print(json.dumps(response1.json(), indent=2))
        
        print("\nTesting benefits endpoint with question...")
        response2 = requests.post(url, json=payload2, headers=headers)
        response2.raise_for_status()
        
        print("Status Code:", response2.status_code)
        print("Response:")
        print(json.dumps(response2.json(), indent=2))
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing benefits endpoint...")
    test_benefits_endpoint()
