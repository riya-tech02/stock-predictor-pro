FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir tensorflow-cpu==2.13.0 && \
    pip install --no-cache-dir fastapi==0.103.1 uvicorn==0.23.2 pydantic==2.3.0 && \
    pip install --no-cache-dir pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.2

# Copy app
COPY . .

# Create dirs
RUN mkdir -p models artifacts logs

# Ensure __init__.py files exist
RUN touch src/__init__.py && \
    touch src/models/__init__.py && \
    touch src/data/__init__.py && \
    touch api/__init__.py

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8000

# Start command
CMD python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}