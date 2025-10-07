"""
Simplified entry point for Render
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Stock Predictor API")

@app.get("/")
async def root():
    return {"message": "API is running!", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)