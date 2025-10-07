"""
Production FastAPI Application for Stock Price Prediction
Includes prediction, explanation, health checks, and monitoring endpoints
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
# ... rest of imports

# Render-specific: Check if models exist
MODEL_PATH = os.getenv("MODEL_PATH", "models/production_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/scaler.pkl")
FEATURE_PATH = os.getenv("FEATURE_PATH", "artifacts/feature_names.json")

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import time
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Custom imports
from src.models.attention_lstm import StockPredictorModel, AttentionLayer
from src.explainability.shap_explainer import ModelExplainer

# ==================== Configuration ====================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Stock Market Predictor API",
    description="Production-grade ML API for stock price movement prediction with explainability",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTIONS_COUNTER = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency'
)
EXPLANATION_LATENCY = Histogram(
    'explanation_latency_seconds',
    'Explanation request latency'
)

# ==================== Pydantic Models ====================

class FeatureInput(BaseModel):
    """
    Input schema for prediction request
    Expects a sequence of feature vectors
    """
    sequence: List[List[float]] = Field(
        ...,
        description="Sequence of feature vectors [time_steps, n_features]",
        example=[[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
    )
    
    @validator('sequence')
    def validate_sequence(cls, v):
        if len(v) == 0:
            raise ValueError("Sequence cannot be empty")
        if len(v) > 200:
            raise ValueError("Sequence length cannot exceed 200")
        # Check all rows have same length
        lengths = [len(row) for row in v]
        if len(set(lengths)) > 1:
            raise ValueError("All feature vectors must have the same length")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    predicted_class: int = Field(..., description="0=Down, 1=Neutral, 2=Up")
    predicted_class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    prediction_id: str
    timestamp: str
    latency_ms: float


class ExplanationRequest(BaseModel):
    """Request schema for explanation endpoint"""
    sequence: List[List[float]]
    prediction_id: Optional[str] = None
    
    @validator('sequence')
    def validate_sequence(cls, v):
        if len(v) == 0:
            raise ValueError("Sequence cannot be empty")
        return v


class ExplanationResponse(BaseModel):
    """Response schema for explanation endpoint"""
    prediction: PredictionResponse
    top_features: List[Dict[str, float]]
    attention_weights: List[float]
    feature_contributions: Dict[str, float]
    explanation_text: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str


# ==================== Global State ====================

class ModelState:
    """Global state for loaded model and dependencies"""
    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[object] = None
        self.explainer: Optional[ModelExplainer] = None
        self.feature_names: Optional[List[str]] = None
        self.sequence_length: int = 60
        self.n_features: int = 40
        self.loaded: bool = False

model_state = ModelState()


# ==================== Startup & Shutdown ====================

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    try:
        print("Loading production model...")
        
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ Model not found at {MODEL_PATH}")
            print("API will start but predictions will fail until model is uploaded")
            model_state.loaded = False
            return
        
        # Load model
        model_state.model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print("✓ Model loaded")
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            model_state.scaler = joblib.load(SCALER_PATH)
            print("✓ Scaler loaded")
        
        # Load feature names
        if os.path.exists(FEATURE_PATH):
            with open(FEATURE_PATH, 'r') as f:
                model_state.feature_names = json.load(f)
            print(f"✓ {len(model_state.feature_names)} features loaded")
            model_state.n_features = len(model_state.feature_names)
        
        model_state.loaded = True
        print("🚀 API ready for predictions!")
        
    except Exception as e:
        print(f"⚠️ Failed to load model: {str(e)}")
        print("API starting in degraded mode - health checks will work")
        model_state.loaded = False




# ==================== Dependency Functions ====================

def get_model_state() -> ModelState:
    """Dependency to check if model is loaded"""
    if not model_state.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    return model_state


# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Market Predictor API v2.0",
        "status": "operational",
        "docs": "/api/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/api/v1/predict",
            "explain": "/api/v1/explain",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy" if model_state.loaded else "unhealthy",
        model_loaded=model_state.loaded,
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(
    request: Request,
    features: FeatureInput,
    state: ModelState = Depends(get_model_state)
):
    """
    Make stock price movement prediction
    
    Returns prediction with confidence scores for Up/Down/Neutral movement
    """
    start_time = time.time()
    prediction_id = f"pred_{int(time.time() * 1000)}"
    
    try:
        # Convert input to numpy array
        sequence = np.array(features.sequence, dtype=np.float32)
        
        # Validate shape
        if sequence.shape[1] != state.n_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {state.n_features} features, got {sequence.shape[1]}"
            )
        
        # Pad or truncate to sequence_length
        if len(sequence) < state.sequence_length:
            # Pad with zeros at the beginning
            padding = np.zeros((state.sequence_length - len(sequence), state.n_features))
            sequence = np.vstack([padding, sequence])
        elif len(sequence) > state.sequence_length:
            # Take last sequence_length steps
            sequence = sequence[-state.sequence_length:]
        
        # Reshape for model input [1, sequence_length, n_features]
        X = sequence.reshape(1, state.sequence_length, state.n_features)
        
        # Make prediction
        with PREDICTION_LATENCY.time():
            predictions = state.model.predict(X, verbose=0)
        
        # Extract results
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        class_names = ['Down', 'Neutral', 'Up']
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log metrics
        PREDICTIONS_COUNTER.labels(endpoint='predict', status='success').inc()
        logger.info(f"Prediction {prediction_id}: {class_names[predicted_class]} ({confidence:.2%})")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_class_name=class_names[predicted_class],
            confidence=confidence,
            probabilities={
                'Down': float(predictions[0][0]),
                'Neutral': float(predictions[0][1]),
                'Up': float(predictions[0][2])
            },
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency_ms
        )
        
    except HTTPException:
        PREDICTIONS_COUNTER.labels(endpoint='predict', status='error').inc()
        raise
    except Exception as e:
        PREDICTIONS_COUNTER.labels(endpoint='predict', status='error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/explain", response_model=ExplanationResponse)
@limiter.limit("20/minute")
async def explain(
    request: Request,
    explain_request: ExplanationRequest,
    state: ModelState = Depends(get_model_state)
):
    """
    Generate explainable prediction with SHAP values and attention weights
    
    Returns detailed explanation of which features influenced the prediction
    """
    start_time = time.time()
    
    try:
        # First, get the prediction
        feature_input = FeatureInput(sequence=explain_request.sequence)
        prediction_response = await predict(request, feature_input, state)
        
        # Prepare sequence
        sequence = np.array(explain_request.sequence, dtype=np.float32)
        if len(sequence) < state.sequence_length:
            padding = np.zeros((state.sequence_length - len(sequence), state.n_features))
            sequence = np.vstack([padding, sequence])
        elif len(sequence) > state.sequence_length:
            sequence = sequence[-state.sequence_length:]
        
        X = sequence.reshape(1, state.sequence_length, state.n_features)
        
        with EXPLANATION_LATENCY.time():
            # Get attention weights
            attention_model = keras.Model(
                inputs=state.model.input,
                outputs=state.model.get_layer('attention').output[1]
            )
            attention_weights = attention_model.predict(X, verbose=0)[0].squeeze()
            
            # Initialize explainer if needed
            if state.explainer is None:
                logger.info("Initializing SHAP explainer...")
                # Use current sample as background (in production, use a proper background dataset)
                state.explainer = ModelExplainer(state.model, state.feature_names)
                state.explainer.initialize_explainer(X)
            
            # Get SHAP values
            explanation = state.explainer.explain_single_prediction(
                X, 
                sample_idx=0,
                class_idx=prediction_response.predicted_class
            )
        
        # Format top features
        top_features = explanation['top_5_features']
        
        # Generate explanation text
        explanation_text = (
            f"The model predicts '{prediction_response.predicted_class_name}' "
            f"with {prediction_response.confidence:.1%} confidence. "
            f"The most influential feature was '{top_features[0]['feature']}'. "
            f"The model focused most on recent time steps (attention weights peak at time step "
            f"{np.argmax(attention_weights)})."
        )
        
        # Calculate feature contributions
        feature_contributions = {
            feat['feature']: feat['shap_value'] 
            for feat in top_features
        }
        
        logger.info(f"Explanation generated in {(time.time() - start_time)*1000:.2f}ms")
        
        return ExplanationResponse(
            prediction=prediction_response,
            top_features=top_features,
            attention_weights=attention_weights.tolist(),
            feature_contributions=feature_contributions,
            explanation_text=explanation_text,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/api/v1/model/info")
async def model_info(state: ModelState = Depends(get_model_state)):
    """Get model information and configuration"""
    return {
        "model_type": "Bi-LSTM with Attention",
        "sequence_length": state.sequence_length,
        "n_features": state.n_features,
        "feature_names": state.feature_names[:10] + ["..."],  # Show first 10
        "classes": {
            0: "Down",
            1: "Neutral",
            2: "Up"
        },
        "loaded": state.loaded,
        "version": "2.0.0"
    }


@app.post("/api/v1/batch/predict")
@limiter.limit("10/minute")
async def batch_predict(
    request: Request,
    batch: List[FeatureInput],
    state: ModelState = Depends(get_model_state)
):
    """
    Batch prediction endpoint for multiple sequences
    Limited to 50 samples per request
    """
    if len(batch) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 50 samples"
        )
    
    results = []
    for idx, features in enumerate(batch):
        try:
            result = await predict(request, features, state)
            results.append({
                "index": idx,
                "prediction": result.dict()
            })
        except Exception as e:
            results.append({
                "index": idx,
                "error": str(e)
            })
    
    return {
        "total": len(batch),
        "successful": sum(1 for r in results if "prediction" in r),
        "failed": sum(1 for r in results if "error" in r),
        "results": results
    }


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=4,
        log_level="info",
        access_log=True
    )
    from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Render"}
