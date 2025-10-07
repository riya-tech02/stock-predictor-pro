from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World", "env": os.getenv("ENV", "unknown")}

@app.get("/health")
def health():
    return {"status": "healthy"}