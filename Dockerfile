FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.23.5 \
    tensorflow-cpu==2.13.0 \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    pydantic==2.3.0 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    joblib==1.3.2 \
    python-dateutil==2.8.2

# Copy application
COPY . /app/

# Create __init__.py files if missing
RUN touch /app/__init__.py && \
    touch /app/api/__init__.py && \
    touch /app/src/__init__.py && \
    touch /app/src/models/__init__.py && \
    touch /app/src/data/__init__.py

# Create directories
RUN mkdir -p /app/models /app/artifacts /app/logs

# Set working directory and Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Simple direct uvicorn command (NO gunicorn)
CMD ["python", "-m", "uvicorn", "simple_main:app", "--host", "0.0.0.0", "--port", "8000"]