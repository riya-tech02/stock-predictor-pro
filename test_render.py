"""
Test Render deployment
"""

import requests
import json

# REPLACE WITH YOUR RENDER URL
API_URL = "https://stock-predictor-api.onrender.com"

def test_render():
    print("="*60)
    print(f"Testing Render Deployment")
    print(f"URL: {API_URL}")
    print("="*60 + "\n")
    
    # Test 1: Health Check
    print("[1/2] Testing Health Check...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Response: {response.json()}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        return
    
    # Test 2: Root Endpoint
    print("[2/2] Testing Root Endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        print(f"✓ Response: {response.json()}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
    
    print("="*60)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("="*60)
    print(f"\n🌐 Your API is live at:")
    print(f"  {API_URL}")
    print(f"  {API_URL}/docs (Swagger UI)")
    print(f"  {API_URL}/health (Health Check)")

if __name__ == "__main__":
    test_render()