"""
Test the Dockerized API
"""

import requests
import json
import time

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"✓ Health: {response.json()}\n")

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info...")
    response = requests.get("http://localhost:8000/api/v1/model/info")
    print(f"✓ Model Info: {json.dumps(response.json(), indent=2)}\n")

def test_prediction():
    """Test prediction endpoint"""
    print("Testing prediction...")
    
    # Create dummy sequence (60 timesteps x 33 features)
    payload = {
        "sequence": [[0.01 * (i + j)] * 33 for i  in range(60)]
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        json=payload
    )
    
    result = response.json()
    print(f"✓ Prediction Result:")
    print(f"  Class: {result['predicted_class_name']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: {result['probabilities']}\n")

if __name__ == "__main__":
    print("="*60)
    print("DOCKER API TEST SUITE")
    print("="*60 + "\n")
    
    # Wait for API to be ready
    print("Waiting for API to start...")
    time.sleep(5)
    
    try:
        test_health()
        test_model_info()
        test_prediction()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")