FROM python:3.9-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy and install requirements ONE BY ONE
COPY requirements.txt .

# Install in specific order to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir tensorflow-cpu==2.13.0
RUN pip install --no-cache-dir fastapi==0.103.1 uvicorn==0.23.2 pydantic==2.3.0
RUN pip install --no-cache-dir pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.2

COPY . .

RUN mkdir -p models artifacts logs

EXPOSE 8000

CMD ["sh", "-c", "cd /app && python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]