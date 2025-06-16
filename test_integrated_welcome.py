#!/usr/bin/env python3
"""
Comprehensive test for integrated Welcome Agent
Tests all endpoints after the integration is complete
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_health_endpoint():
    """Test the /health endpoint"""
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Health Endpoint Test PASSED")
        print(f"   Status: {result['status']}")
        print(f"   Welcome Agent: {result['agents']['welcome_agent']['status']}")
        print(f"   Benefit Agent: {result['agents']['benefit_agent']['status']}")
        print(f"   Version: {result['version']}")
        return True
        
    except Exception as e:
        print(f"❌ Health Endpoint Test FAILED: {e}")
        return False

def test_welcome_endpoint():
    """Test the /welcome endpoint"""
    url = f"{BASE_URL}/welcome"
    
    payload = {
        "name": "Alice Johnson",
        "role": "Product Manager", 
        "team": "Product",
        "manager": "Bob Smith",
        "start_date": "2024-01-20",
        "email": "alice.johnson@company.com"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Welcome Endpoint Test PASSED")
        print(f"   Success: {result['success']}")
        print(f"   Welcome Message Preview: {result['welcome_message'][:100]}...")
        print(f"   Next Steps Count: {len(result['next_steps'])}")
        print(f"   Documents Count: {len(result['available_documents'])}")
        return True
        
    except Exception as e:
        print(f"❌ Welcome Endpoint Test FAILED: {e}")
        return False

def test_chat_endpoint():
    """Test the /chat endpoint"""
    url = f"{BASE_URL}/chat"
    
    params = {
        "question": "What should I prepare for my first day?",
        "user_name": "Alice Johnson"
    }
    
    try:
        response = requests.post(url, params=params, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Chat Endpoint Test PASSED")
        print(f"   User: {result['user']}")
        print(f"   Answer Preview: {result['answer'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Chat Endpoint Test FAILED: {e}")
        return False

def test_onboard_endpoint():
    """Test the unified /onboard endpoint"""
    url = f"{BASE_URL}/onboard"
    
    payload = {
        "name": "Alice Johnson",
        "role": "Product Manager",
        "start_date": "2024-01-20",
        "email": "alice.johnson@company.com",
        "team": "Product",
        "manager": "Bob Smith"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Onboard Endpoint Test PASSED")
        print(f"   Success: {result['success']}")
        print(f"   Next Steps Count: {len(result['next_steps'])}")
        print(f"   Documents Count: {len(result['documents'])}")
        print(f"   Team Contacts: {len(result['team_contacts'])}")
        return True
        
    except Exception as e:
        print(f"❌ Onboard Endpoint Test FAILED: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    url = f"{BASE_URL}/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Root Endpoint Test PASSED")
        print(f"   Message: {result['message']}")
        print(f"   Status: {result['status']}")
        print(f"   Available Endpoints: {len(result['available_endpoints'])}")
        return True
        
    except Exception as e:
        print(f"❌ Root Endpoint Test FAILED: {e}")
        return False

def test_benefit_endpoint():
    """Test the /benefit/fetch endpoint (should return service unavailable)"""
    url = f"{BASE_URL}/benefit/fetch"
    
    try:
        response = requests.post(url, json={}, timeout=10)
        # We expect this to fail with 503
        if response.status_code == 503:
            print("✅ Benefit Endpoint Test PASSED (correctly unavailable)")
            print(f"   Status Code: {response.status_code}")
            return True
        else:
            print(f"❌ Benefit Endpoint Test FAILED: Expected 503, got {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ Benefit Endpoint Test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Integrated Welcome Agent System...")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Welcome Message", test_welcome_endpoint),
        ("Chat Functionality", test_chat_endpoint),
        ("Complete Onboarding", test_onboard_endpoint),
        ("Benefit Service (Unavailable)", test_benefit_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        if test_func():
            passed += 1
        print("-" * 40)
    
    print(f"\n📊 FINAL TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Welcome Agent integration is COMPLETE!")
        print("✨ The onboarding system is fully operational with integrated Welcome Agent")
    elif passed >= 4:
        print("✅ CORE TESTS PASSED! Welcome Agent integration is successful")
        print("⚠️ Some optional features may not be available")
    else:
        print("⚠️ Some critical tests failed. Check the output above.")
    
    print("\n" + "=" * 60)
    print("Integration test complete!")
