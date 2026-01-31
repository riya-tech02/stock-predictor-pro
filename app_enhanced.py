"""
Enhanced Stock Predictor with Advanced Features
Production-Ready Version with Technical Analysis & Risk Management
"""

import sys
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import json
import time
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Stock Market Predictor",
    version="4.0.0",
    description="AI-powered stock prediction with technical analysis and risk assessment"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Enhanced Models ====================

class StockRequest(BaseModel):
    ticker: str
    use_real_data: bool = True
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError('Invalid ticker symbol')
        return v.upper().strip()

class TechnicalIndicators(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    volume_trend: Optional[str] = None
    volatility: Optional[float] = None

class RiskMetrics(BaseModel):
    confidence_level: str  # High, Medium, Low
    volatility_rating: str  # High, Medium, Low
    trend_strength: float  # 0-100
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

class MarketContext(BaseModel):
    sector_trend: Optional[str] = None
    market_sentiment: Optional[str] = None
    trading_volume_status: str
    price_momentum: str

class PredictionResponse(BaseModel):
    # Core prediction
    predicted_class: int
    predicted_class_name: str
    confidence: float
    probabilities: Dict[str, float]
    
    # Stock data
    ticker: str
    current_price: Optional[float] = None
    change_percent: Optional[float] = None
    change_amount: Optional[float] = None
    
    # Technical analysis
    technical_indicators: TechnicalIndicators
    risk_metrics: RiskMetrics
    market_context: MarketContext
    
    # Target predictions
    target_price_1d: Optional[float] = None
    target_price_1w: Optional[float] = None
    stop_loss_suggestion: Optional[float] = None
    
    # Meta
    timestamp: str
    model_loaded: bool
    data_source: str
    data_quality: str  # Excellent, Good, Fair, Estimated
    recommendation: str  # Strong Buy, Buy, Hold, Sell, Strong Sell

class HistoricalDataRequest(BaseModel):
    ticker: str
    days: int = Field(default=30, ge=1, le=365)

class ModelState:
    def __init__(self):
        self.model = None
        self.loaded = False
        self.load_time = None

model_state = ModelState()

# Enhanced caching with metadata
stock_cache = {}
CACHE_DURATION = 300  # 5 minutes
prediction_history = defaultdict(list)
MAX_HISTORY = 100

# Rate limiting
request_timestamps = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # 1 minute
MAX_REQUESTS_PER_WINDOW = 10

# ==================== Rate Limiting ====================

def check_rate_limit(client_id: str = "default") -> bool:
    """Check if client has exceeded rate limit"""
    current_time = time.time()
    timestamps = request_timestamps[client_id]
    
    # Remove old timestamps
    timestamps[:] = [ts for ts in timestamps if current_time - ts < RATE_LIMIT_WINDOW]
    
    if len(timestamps) >= MAX_REQUESTS_PER_WINDOW:
        return False
    
    timestamps.append(current_time)
    return True

# ==================== Enhanced Stock Data Fetcher ====================

def calculate_technical_indicators(hist) -> TechnicalIndicators:
    """Calculate comprehensive technical indicators"""
    try:
        import pandas as pd
        
        # RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = hist['Close'].ewm(span=12).mean()
        ema_26 = hist['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = hist['Close'].rolling(20).mean()
        std_20 = hist['Close'].rolling(20).std()
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        
        # SMAs
        sma_50 = hist['Close'].rolling(50).mean()
        
        # Volume trend
        vol_sma = hist['Volume'].rolling(20).mean()
        recent_vol = hist['Volume'].iloc[-5:].mean()
        volume_trend = "High" if recent_vol > vol_sma.iloc[-1] * 1.2 else "Normal" if recent_vol > vol_sma.iloc[-1] * 0.8 else "Low"
        
        # Volatility
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        return TechnicalIndicators(
            rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            macd=float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
            macd_signal=float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None,
            bb_upper=float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
            bb_lower=float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
            sma_20=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
            sma_50=float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
            ema_12=float(ema_12.iloc[-1]),
            ema_26=float(ema_26.iloc[-1]),
            volume_trend=volume_trend,
            volatility=float(volatility) if not pd.isna(volatility) else None
        )
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return TechnicalIndicators()

def calculate_risk_metrics(hist, current_price: float, confidence: float) -> RiskMetrics:
    """Calculate risk management metrics"""
    try:
        # Support and resistance (simplified)
        recent_high = float(hist['High'].iloc[-20:].max())
        recent_low = float(hist['Low'].iloc[-20:].min())
        
        # Volatility rating
        returns = hist['Close'].pct_change()
        volatility = returns.std()
        vol_rating = "High" if volatility > 0.03 else "Medium" if volatility > 0.015 else "Low"
        
        # Trend strength based on moving averages
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            trend_diff = abs((sma_20 - sma_50) / sma_50) * 100
            trend_strength = min(trend_diff * 10, 100)
        else:
            trend_strength = 50.0
        
        # Confidence level
        conf_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
        
        # Risk-reward ratio
        distance_to_resistance = (recent_high - current_price) / current_price
        distance_to_support = (current_price - recent_low) / current_price
        risk_reward = distance_to_resistance / distance_to_support if distance_to_support > 0 else None
        
        return RiskMetrics(
            confidence_level=conf_level,
            volatility_rating=vol_rating,
            trend_strength=float(trend_strength),
            support_level=recent_low,
            resistance_level=recent_high,
            risk_reward_ratio=float(risk_reward) if risk_reward else None
        )
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return RiskMetrics(
            confidence_level="Medium",
            volatility_rating="Medium",
            trend_strength=50.0
        )

def calculate_market_context(hist) -> MarketContext:
    """Determine market context and sentiment"""
    try:
        # Price momentum
        returns_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-6] - 1) * 100
        momentum = "Strong Bullish" if returns_5d > 3 else "Bullish" if returns_5d > 1 else "Bearish" if returns_5d < -1 else "Strong Bearish" if returns_5d < -3 else "Neutral"
        
        # Volume status
        vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
        current_vol = hist['Volume'].iloc[-1]
        vol_status = "Above Average" if current_vol > vol_avg * 1.2 else "Below Average" if current_vol < vol_avg * 0.8 else "Average"
        
        return MarketContext(
            trading_volume_status=vol_status,
            price_momentum=momentum,
            market_sentiment="Bullish" if returns_5d > 0 else "Bearish"
        )
    except Exception as e:
        logger.error(f"Error calculating market context: {e}")
        return MarketContext(
            trading_volume_status="Average",
            price_momentum="Neutral",
            market_sentiment="Neutral"
        )

def calculate_price_targets(current_price: float, predicted_class: int, volatility: float, confidence: float) -> Tuple[float, float, float]:
    """Calculate price targets and stop loss"""
    # Adjust targets based on volatility and confidence
    vol_multiplier = 1 + (volatility / 100) if volatility else 1.02
    
    if predicted_class == 2:  # Up
        target_1d = current_price * (1 + 0.015 * vol_multiplier * confidence)
        target_1w = current_price * (1 + 0.05 * vol_multiplier * confidence)
        stop_loss = current_price * 0.97
    elif predicted_class == 0:  # Down
        target_1d = current_price * (1 - 0.015 * vol_multiplier * confidence)
        target_1w = current_price * (1 - 0.05 * vol_multiplier * confidence)
        stop_loss = current_price * 1.03
    else:  # Neutral
        target_1d = current_price
        target_1w = current_price
        stop_loss = current_price * 0.98
    
    return target_1d, target_1w, stop_loss

def generate_recommendation(predicted_class: int, confidence: float, rsi: Optional[float], 
                           volatility_rating: str) -> str:
    """Generate trading recommendation"""
    if predicted_class == 2 and confidence > 0.7 and (not rsi or rsi < 70):
        return "Strong Buy" if volatility_rating != "High" else "Buy"
    elif predicted_class == 2 and confidence > 0.5:
        return "Buy"
    elif predicted_class == 0 and confidence > 0.7 and (not rsi or rsi > 30):
        return "Strong Sell" if volatility_rating != "High" else "Sell"
    elif predicted_class == 0 and confidence > 0.5:
        return "Sell"
    else:
        return "Hold"

def fetch_real_stock_data(ticker: str) -> Tuple:
    """Enhanced stock data fetcher with comprehensive analysis"""
    
    # Check cache first
    cache_key = ticker
    if cache_key in stock_cache:
        cached_data = stock_cache[cache_key]
        if time.time() - cached_data['timestamp'] < CACHE_DURATION:
            logger.info(f"✓ Using cached data for {ticker}")
            return (
                cached_data['sequence'], 
                cached_data['price'], 
                cached_data['change'],
                cached_data['change_amount'],
                cached_data['technical_indicators'],
                cached_data['hist'],
                None,
                "Good"
            )
    
    try:
        import yfinance as yf
        import pandas as pd
        
        logger.info(f"Fetching fresh data for {ticker}...")
        time.sleep(0.5)  # Rate limit protection
        
        # Fetch more data for better technical analysis
        hist = yf.download(ticker, period="6mo", progress=False, show_errors=False)
        
        if len(hist) == 0:
            raise ValueError(f"No data available for {ticker}")
        
        # Current price and change
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100
        
        # Calculate comprehensive technical indicators
        technical_indicators = calculate_technical_indicators(hist)
        
        # Calculate base features for model
        hist['Returns'] = hist['Close'].pct_change()
        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['Vol_Ratio'] = hist['Volume'] / hist['Volume'].rolling(20).mean()
        
        # Build sequence for model
        sequence = []
        for i in range(min(60, len(hist))):
            idx = -(60-i)
            row = hist.iloc[idx]
            
            features = [
                float(row['Returns']) if not pd.isna(row['Returns']) else 0.0,
                float((row['High'] - row['Low']) / row['Close']) if row['Close'] != 0 else 0.0,
                float(row['Close'] / row['SMA_20']) if not pd.isna(row['SMA_20']) and row['SMA_20'] != 0 else 1.0,
                float(technical_indicators.rsi / 100.0) if technical_indicators.rsi else 0.5,
                float(row['Vol_Ratio']) if not pd.isna(row['Vol_Ratio']) else 1.0,
            ]
            features.extend([0.0] * (33 - len(features)))
            sequence.append(features[:33])
        
        while len(sequence) < 60:
            sequence.insert(0, [0.0] * 33)
        
        # Cache the result
        stock_cache[cache_key] = {
            'sequence': sequence,
            'price': current_price,
            'change': change_percent,
            'change_amount': change_amount,
            'technical_indicators': technical_indicators,
            'hist': hist,
            'timestamp': time.time()
        }
        
        logger.info(f"✓ {ticker}: ${current_price:.2f} ({change_percent:+.2f}%)")
        
        return sequence, current_price, change_percent, change_amount, technical_indicators, hist, None, "Excellent"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error fetching {ticker}: {error_msg}")
        
        # Generate realistic demo data
        return generate_demo_data(ticker, error_msg)

def generate_demo_data(ticker: str, error_msg: str) -> Tuple:
    """Generate realistic demo data when API fails"""
    import pandas as pd
    
    demo_prices = {
        'AAPL': 178.50, 'TSLA': 245.30, 'GOOGL': 142.80,
        'MSFT': 415.20, 'AMZN': 178.90, 'META': 485.60,
        'NVDA': 875.40, 'SPY': 512.30
    }
    
    demo_price = demo_prices.get(ticker, 150.00)
    demo_change = np.random.uniform(-2.0, 2.0)
    demo_change_amount = demo_price * (demo_change / 100)
    
    # Generate realistic sequence
    base_trend = np.linspace(-0.1, 0.1, 60)
    noise = np.random.randn(60, 33) * 0.05
    sequence = (base_trend[:, np.newaxis] + noise).tolist()
    
    # Create demo technical indicators
    technical_indicators = TechnicalIndicators(
        rsi=np.random.uniform(40, 60),
        volume_trend="Normal",
        volatility=np.random.uniform(15, 30)
    )
    
    # Create minimal hist dataframe
    dates = pd.date_range(end=datetime.now(), periods=60)
    hist = pd.DataFrame({
        'Close': np.linspace(demo_price * 0.9, demo_price, 60),
        'High': np.linspace(demo_price * 0.92, demo_price * 1.02, 60),
        'Low': np.linspace(demo_price * 0.88, demo_price * 0.98, 60),
        'Volume': np.random.randint(1000000, 10000000, 60)
    }, index=dates)
    
    return (sequence, demo_price, demo_change, demo_change_amount, 
            technical_indicators, hist, "Rate limited - using estimated data", "Estimated")

# ==================== Startup ====================

@app.on_event("startup")
async def startup():
    logger.info("="*60)
    logger.info("🚀 Advanced Stock Predictor Starting...")
    logger.info("="*60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        MODEL_PATH = "models/production_model.h5"
        
        if os.path.exists(MODEL_PATH):
            from tensorflow.keras import layers
            
            class AttentionLayerKeras(layers.Layer):
                def __init__(self, units=128, **kwargs):
                    super(AttentionLayerKeras, self).__init__(**kwargs)
                    self.units = units
                    self.W = layers.Dense(units, activation='tanh')
                    self.V = layers.Dense(1)
                    
                def call(self, inputs):
                    score = self.V(self.W(inputs))
                    attention_weights = tf.nn.softmax(score, axis=1)
                    context_vector = attention_weights * inputs
                    context_vector = tf.reduce_sum(context_vector, axis=1)
                    return context_vector, attention_weights
                
                def get_config(self):
                    config = super().get_config()
                    config.update({"units": self.units})
                    return config
            
            model_state.model = keras.models.load_model(
                MODEL_PATH,
                custom_objects={'AttentionLayer': AttentionLayerKeras}
            )
            model_state.loaded = True
            model_state.load_time = datetime.now()
            logger.info("✅ Model loaded successfully!")
        else:
            logger.warning("⚠️  Running in demo mode - model file not found")
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")

# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve enhanced frontend"""
    # This would be the enhanced HTML - keeping original for now
    # You can replace with the enhanced version below
    from pathlib import Path
    html_path = Path("templates/index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Stock Predictor API</h1><p>Use /docs for API documentation</p>"

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_state.loaded,
        "model_load_time": model_state.load_time.isoformat() if model_state.load_time else None,
        "cache_size": len(stock_cache),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/stats")
def get_stats():
    """Get system statistics"""
    return {
        "total_predictions": sum(len(v) for v in prediction_history.values()),
        "unique_tickers": len(prediction_history),
        "cached_tickers": len(stock_cache),
        "model_status": "loaded" if model_state.loaded else "demo_mode",
        "uptime": (datetime.now() - model_state.load_time).total_seconds() if model_state.load_time else 0
    }

@app.post("/api/v1/predict/stock", response_model=PredictionResponse)
def predict_stock(request: StockRequest):
    """Enhanced stock prediction endpoint"""
    
    # Rate limiting
    if not check_rate_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait.")
    
    ticker = request.ticker
    
    try:
        # Fetch comprehensive data
        (sequence, current_price, change_percent, change_amount, 
         technical_indicators, hist, error, data_quality) = fetch_real_stock_data(ticker)
        
        data_source = "Live market data" if not error else error
        
        sequence_array = np.array(sequence, dtype=np.float32)
        
        # Make prediction
        if model_state.loaded:
            X = sequence_array.reshape(1, 60, 33)
            predictions = model_state.model.predict(X, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])
            probabilities = {
                'Down': float(predictions[0][0]),
                'Neutral': float(predictions[0][1]),
                'Up': float(predictions[0][2])
            }
        else:
            # Demo mode with smart heuristics
            input_mean = float(np.mean(sequence_array))
            rsi_factor = (technical_indicators.rsi - 50) / 50 if technical_indicators.rsi else 0
            
            base_up = 0.33 + (input_mean * 2) + (rsi_factor * 0.2)
            base_down = 0.33 - (input_mean * 2) - (rsi_factor * 0.2)
            base_neutral = 1 - base_up - base_down
            
            probs = [max(0.1, min(0.8, base_down)), 
                    max(0.1, min(0.6, base_neutral)),
                    max(0.1, min(0.8, base_up))]
            
            # Normalize
            total = sum(probs)
            probs = [p/total for p in probs]
            
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            probabilities = {
                'Down': float(probs[0]),
                'Neutral': float(probs[1]),
                'Up': float(probs[2])
            }
        
        # Calculate risk metrics and market context
        risk_metrics = calculate_risk_metrics(hist, current_price, confidence)
        market_context = calculate_market_context(hist)
        
        # Calculate price targets
        target_1d, target_1w, stop_loss = calculate_price_targets(
            current_price, predicted_class, 
            technical_indicators.volatility or 20, confidence
        )
        
        # Generate recommendation
        recommendation = generate_recommendation(
            predicted_class, confidence,
            technical_indicators.rsi,
            risk_metrics.volatility_rating
        )
        
        # Store in history
        prediction_history[ticker].append({
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': predicted_class,
            'confidence': confidence,
            'price': current_price
        })
        
        # Keep only recent history
        if len(prediction_history[ticker]) > MAX_HISTORY:
            prediction_history[ticker] = prediction_history[ticker][-MAX_HISTORY:]
        
        response = PredictionResponse(
            predicted_class=predicted_class,
            predicted_class_name=['Down', 'Neutral', 'Up'][predicted_class],
            confidence=confidence,
            probabilities=probabilities,
            ticker=ticker,
            current_price=current_price,
            change_percent=change_percent,
            change_amount=change_amount,
            technical_indicators=technical_indicators,
            risk_metrics=risk_metrics,
            market_context=market_context,
            target_price_1d=target_1d,
            target_price_1w=target_1w,
            stop_loss_suggestion=stop_loss,
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=model_state.loaded,
            data_source=data_source,
            data_quality=data_quality,
            recommendation=recommendation
        )
        
        logger.info(f"Prediction for {ticker}: {response.predicted_class_name} ({confidence:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/v1/history/{ticker}")
def get_prediction_history(ticker: str, limit: int = 20):
    """Get prediction history for a ticker"""
    ticker = ticker.upper()
    history = prediction_history.get(ticker, [])
    return {
        "ticker": ticker,
        "count": len(history),
        "predictions": history[-limit:]
    }

@app.post("/api/v1/batch-predict")
async def batch_predict(tickers: List[str]):
    """Batch prediction for multiple tickers"""
    if len(tickers) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers per batch")
    
    results = []
    for ticker in tickers:
        try:
            request = StockRequest(ticker=ticker, use_real_data=True)
            result = predict_stock(request)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch for {ticker}: {e}")
            results.append({"ticker": ticker, "error": str(e)})
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")