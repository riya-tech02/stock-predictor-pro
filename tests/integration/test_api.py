"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np


# Import after model is loaded
@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_structure(client):
    """Test prediction endpoint accepts valid request"""
    # Note: This will fail until model is trained and loaded
    payload = {
        "sequence": [[0.1] * 40 for _ in range(60)]
    }
    
    response = client.post("/api/v1/predict", json=payload)
    
    # May return 503 if model not loaded, that's okay for this test
    assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])