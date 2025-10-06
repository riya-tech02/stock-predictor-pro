"""
Unit tests for model architecture
"""

import pytest
import numpy as np
from src.models.attention_lstm import StockPredictorModel, AttentionLayer


def test_attention_layer():
    """Test custom attention layer"""
    attention = AttentionLayer(units=64)
    dummy_input = np.random.randn(32, 60, 40)
    
    context, weights = attention(dummy_input)
    
    assert context.shape == (32, 40)
    assert weights.shape == (32, 60, 1)


def test_model_building():
    """Test model can be built"""
    model = StockPredictorModel(
        sequence_length=60,
        n_features=40,
        lstm_units=[128, 64]
    )
    
    model.build_model()
    
    assert model.model is not None
    assert model.attention_model is not None


def test_model_prediction():
    """Test model can make predictions"""
    model = StockPredictorModel(sequence_length=60, n_features=40)
    model.build_model()
    
    dummy_data = np.random.randn(10, 60, 40)
    predictions = model.model.predict(dummy_data, verbose=0)
    
    assert predictions.shape == (10, 3)
    assert np.allclose(predictions.sum(axis=1), 1.0)  # Softmax check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])