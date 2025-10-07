import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from datetime import datetime

app = FastAPI(title="Stock Predictor API")

# Your existing code from api/main.py goes here
# Just paste the contents of api/main.py into this file

@app.get("/")
def root():
    return {
        "message": "Stock Predictor API v2.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": False,
        "version": "2.0.0"
    }

# Add your other endpoints...