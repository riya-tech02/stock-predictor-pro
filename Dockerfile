# ============================================
# Production Dockerfile - Stock Predictor
# Zero dependency conflicts
# ============================================

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies in correct order to avoid conflicts
RUN pip install --upgrade pip && \
    # Install typing-extensions first (critical for TensorFlow compatibility)
    pip install "typing-extensions>=3.6.6,<4.6.0" && \
    # Install TensorFlow and ML stack
    pip install tensorflow-cpu==2.13.0 numpy==1.23.5 && \
    # Install remaining dependencies
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models artifacts logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]