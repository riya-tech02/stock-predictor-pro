"""
Run the FastAPI server locally
"""

import uvicorn
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("Starting Stock Predictor API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs: http://localhost:8000/api/docs")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )