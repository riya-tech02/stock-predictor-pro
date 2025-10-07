"""
Download model from external storage on Render startup
"""

import os
import requests

def download_file(url, destination):
    """Download file from URL"""
    print(f"Downloading {destination}...")
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded {destination}")

if __name__ == "__main__":
    # Replace with your Google Drive or S3 URLs
    MODEL_URL = "YOUR_MODEL_URL_HERE"
    SCALER_URL = "YOUR_SCALER_URL_HERE"
    FEATURES_URL = "YOUR_FEATURES_URL_HERE"
    
    download_file(MODEL_URL, "models/production_model.h5")
    download_file(SCALER_URL, "artifacts/scaler.pkl")
    download_file(FEATURES_URL, "artifacts/feature_names.json")