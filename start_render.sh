#!/bin/bash

# Render startup script
echo "Starting Stock Predictor API on Render..."

# Check if models exist
if [ ! -f "models/production_model.h5" ]; then
    echo "⚠️  Warning: Model files not found!"
    echo "Models should be committed to git or uploaded to external storage"
fi

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}