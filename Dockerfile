FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir tensorflow-cpu==2.13.0 && \
    pip install --no-cache-dir typing-extensions==4.5.0 && \
    pip install --no-cache-dir fastapi==0.103.1 uvicorn==0.23.2 pydantic==2.3.0 && \
    pip install --no-cache-dir scikit-learn==1.3.0 joblib==1.3.2 && \
    pip install --no-cache-dir yfinance==0.2.28 pandas==2.0.3

COPY . .

RUN mkdir -p models artifacts logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]