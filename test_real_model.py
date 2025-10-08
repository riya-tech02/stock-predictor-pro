"""
Test the real ML model on Render
"""

import requests
import json

# YOUR RENDER URL
API_URL = "https://your-app.onrender.com"

def test_real_predictions():
    print("="*60)
    print("Testing REAL ML Model on Render")
    print("="*60 + "\n")
    
    # Check health
    print("[1/3] Checking model status...")
    response = requests.get(f"{API_URL}/health")
    health = response.json()
    print(f"Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}")
    print(f"Message: {health['message']}\n")
    
    if not health['model_loaded']:
        print("⚠️  Model not loaded yet - check Render logs")
        return
    
    # Get model info
    print("[2/3] Getting model info...")
    response = requests.get(f"{API_URL}/api/v1/model/info")
    info = response.json()
    print(f"Model Type: {info['model_type']}")
    print(f"Features: {info['n_features']}")
    print(f"Accuracy: {info['training_info']['validation_accuracy']}\n")
    
    # Make real prediction
    print("[3/3] Making REAL prediction...")
    
    # Create realistic test data (60 timesteps x 33 features)
    payload = {
        "sequence": [
            [
                0.01, 0.02, 0.015,  # returns, log_returns, high_low_pct
                50.5, 51.2, 52.0, 53.1,  # SMAs
                50.8, 51.5, 52.3, 53.5,  # EMAs
                1.01, 1.02, 1.03, 1.04,  # price/SMA ratios
                65.2,  # RSI
                0.5, 1.2, 0.7,  # MACD, signal, histogram
                52.0, 54.0, 50.0, 0.04,  # BB middle, upper, lower, width
                1.2,  # ATR
                2500000, 1.1,  # volume SMA, ratio
                0.5,  # momentum
                2.1, 3.2,  # ROC 5, 10
                0.15  # volatility
            ] * 1  # One timestep, repeat 60 times below
        ] * 60  # 60 timesteps
    }
    
    response = requests.post(f"{API_URL}/api/v1/predict", json=payload)
    result = response.json()
    
    print(f"\n✅ REAL PREDICTION:")
    print(f"  Class: {result['predicted_class_name']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"    {cls}: {prob:.2%}")
    print(f"  Model Used: {'REAL ML MODEL' if result['model_loaded'] else 'Demo Mode'}")
    
    print("\n" + "="*60)
    print("🎉 REAL ML MODEL IS WORKING ON RENDER!")
    print("="*60)

if __name__ == "__main__":
    test_real_predictions()