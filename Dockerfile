FROM python:3.9-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    pydantic==2.3.0 \
    numpy==1.23.5

# Copy app
COPY app.py .

# Create dirs for future model files
RUN mkdir -p models artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]