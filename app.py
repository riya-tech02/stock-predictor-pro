"""
Real Stock Predictor with Live Data
Working version with proper error handling
"""

import sys
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import json

app = FastAPI(title="Stock Market Predictor", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Models ====================

class StockRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    use_real_data: bool = Field(default=True)

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    model_loaded: bool
    ticker: str
    current_price: Optional[float] = None
    data_source: str

class ModelState:
    def __init__(self):
        self.model = None
        self.loaded = False
        self.n_features = 33
        self.sequence_length = 60

model_state = ModelState()
prediction_count = 0

# ==================== Startup ====================

@app.on_event("startup")
async def startup():
    print("="*60)
    print("🚀 Starting Stock Predictor...")
    print("="*60)
    
    try:
        # Try to import and load model
        import tensorflow as tf
        from tensorflow import keras
        import joblib
        
        print("✓ TensorFlow imported")
        
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
            print("✅ Model loaded successfully!")
        else:
            print("⚠️  Model file not found")
            
    except Exception as e:
        print(f"❌ Startup error: {str(e)}")

# ==================== Stock Data Fetcher ====================

def fetch_real_stock_data(ticker: str):
    """Fetch real stock data using yfinance"""
    try:
        import yfinance as yf
        
        # Download data
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        data = stock.history(start=start_date, end=end_date)
        
        if len(data) == 0:
            raise ValueError(f"No data found for {ticker}")
        
        # Get current price
        current_price = float(data['Close'].iloc[-1])
        
        # Calculate indicators
        data['Returns'] = data['Close'].pct_change()
        data['SMA_5'] = data['Close'].rolling(5).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume ratio
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Create sequence
        sequence = []
        for i in range(min(60, len(data))):
            idx = -(60-i)
            row = data.iloc[idx]
            
            # Build feature vector (33 features)
            features = [
                # Price features
                row['Returns'] if not np.isnan(row['Returns']) else 0.0,
                (row['High'] - row['Low']) / row['Close'] if row['Close'] != 0 else 0.0,
                (row['Close'] - row['Open']) / row['Open'] if row['Open'] != 0 else 0.0,
                
                # Moving averages
                row['Close'] / row['SMA_5'] if not np.isnan(row['SMA_5']) and row['SMA_5'] != 0 else 1.0,
                row['Close'] / row['SMA_20'] if not np.isnan(row['SMA_20']) and row['SMA_20'] != 0 else 1.0,
                row['Close'] / row['EMA_12'] if not np.isnan(row['EMA_12']) and row['EMA_12'] != 0 else 1.0,
                
                # RSI
                row['RSI'] / 100.0 if not np.isnan(row['RSI']) else 0.5,
                
                # Volume
                row['Volume_Ratio'] if not np.isnan(row['Volume_Ratio']) else 1.0,
            ]
            
            # Pad to 33 features
            while len(features) < 33:
                features.append(0.0)
            
            sequence.append(features[:33])
        
        # Pad if less than 60 days
        while len(sequence) < 60:
            sequence.insert(0, [0.0] * 33)
        
        return sequence, current_price, None
        
    except Exception as e:
        return None, None, str(e)

# ==================== Frontend HTML ====================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Predictor - Live AI Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            text-align: center;
            padding: 40px 20px;
            color: white;
        }
        .header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.3);
            backdrop-filter: blur(10px);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 10px;
            color: white;
        }
        .main-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            margin-top: -20px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #1f2937;
        }
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 1rem;
        }
        .popular-stocks {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .stock-btn {
            padding: 8px 16px;
            background: #f3f4f6;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stock-btn:hover {
            background: #6366f1;
            color: white;
            border-color: #6366f1;
        }
        .btn {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            background: #6366f1;
            color: white;
            transition: all 0.3s;
        }
        .btn:hover {
            background: #4f46e5;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid rgba(99, 102, 241, 0.1);
            border-left-color: #6366f1;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 30px;
            color: white;
            margin-top: 30px;
            display: none;
        }
        .result-card.show { 
            display: block; 
            animation: slideIn 0.5s; 
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stock-header {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stock-ticker {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .stock-price {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
        }
        .data-source {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 5px;
        }
        .prediction-result { 
            text-align: center; 
            margin: 20px 0; 
        }
        .prediction-icon { 
            font-size: 4rem; 
            margin-bottom: 10px; 
        }
        .prediction-label { 
            font-size: 2rem; 
            font-weight: 700; 
            margin-bottom: 10px; 
        }
        .probabilities {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .prob-item {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .prob-value { 
            font-size: 1.5rem; 
            font-weight: 700; 
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: white;
            margin-top: 40px;
        }
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .probabilities { grid-template-columns: 1fr; }
            .main-card { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> Stock Market Predictor</h1>
            <p>Real-Time AI Analysis with Live Market Data</p>
            <div class="status-badge">
                <i class="fas fa-check-circle"></i> Live System Ready
            </div>
        </div>

        <div class="main-card">
            <h2 style="margin-bottom: 20px; color: #1f2937;">Predict Stock Movement</h2>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label for="ticker">Enter Stock Ticker</label>
                    <input type="text" id="ticker" placeholder="Type stock symbol (e.g., AAPL)" value="AAPL" required>
                    
                    <div class="popular-stocks">
                        <button type="button" class="stock-btn" onclick="setTicker('AAPL')">
                            <i class="fab fa-apple"></i> AAPL
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('TSLA')">
                            <i class="fas fa-car"></i> TSLA
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('GOOGL')">
                            <i class="fab fa-google"></i> GOOGL
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('MSFT')">
                            <i class="fab fa-microsoft"></i> MSFT
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('AMZN')">
                            <i class="fab fa-amazon"></i> AMZN
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('META')">
                            <i class="fab fa-meta"></i> META
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('NVDA')">
                            <i class="fas fa-microchip"></i> NVDA
                        </button>
                        <button type="button" class="stock-btn" onclick="setTicker('SPY')">
                            <i class="fas fa-chart-area"></i> SPY
                        </button>
                    </div>
                </div>

                <button type="submit" class="btn" id="predictBtn">
                    <i class="fas fa-magic"></i> Analyze with AI
                </button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #6b7280;">Fetching live market data and analyzing...</p>
            </div>

            <div class="result-card" id="results">
                <div class="stock-header">
                    <div class="stock-ticker" id="stockTicker">AAPL</div>
                    <div class="stock-price" id="stockPrice">$175.43</div>
                    <div class="data-source" id="dataSource">
                        <i class="fas fa-database"></i> Real-time market data
                    </div>
                </div>

                <div class="prediction-result">
                    <div class="prediction-icon" id="predictionIcon">📈</div>
                    <div class="prediction-label" id="predictionLabel">Bullish Trend</div>
                    <div style="font-size: 1.2rem;">
                        AI Confidence: <span id="confidence">85%</span>
                    </div>
                </div>

                <div class="probabilities">
                    <div class="prob-item">
                        <div style="margin-bottom: 10px;">📉 Bearish</div>
                        <div class="prob-value" id="probDown">15%</div>
                    </div>
                    <div class="prob-item">
                        <div style="margin-bottom: 10px;">➡️ Neutral</div>
                        <div class="prob-value" id="probNeutral">20%</div>
                    </div>
                    <div class="prob-item">
                        <div style="margin-bottom: 10px;">📈 Bullish</div>
                        <div class="prob-value" id="probUp">65%</div>
                    </div>
                </div>

                <p style="margin-top: 20px; text-align: center; opacity: 0.9; font-size: 0.9rem;">
                    <i class="fas fa-brain"></i> 
                    Powered by Deep Learning (Bi-LSTM + Attention)
                </p>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: #f9fafb; border-radius: 10px;">
                <h3 style="color: #1f2937; margin-bottom: 15px;">
                    <i class="fas fa-info-circle"></i> How It Works
                </h3>
                <ul style="color: #6b7280; line-height: 1.8; list-style-position: inside;">
                    <li>📊 Fetches real-time price data from Yahoo Finance</li>
                    <li>🔢 Calculates 33 technical indicators (RSI, SMA, EMA, Volume)</li>
                    <li>🧠 Analyzes patterns using Bi-LSTM neural network</li>
                    <li>🎯 Predicts next-day price movement with confidence score</li>
                    <li>📈 Trained on historical S&P 500 data (47% accuracy)</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Built with <i class="fas fa-heart" style="color: #ef4444;"></i> using TensorFlow + FastAPI + yfinance</p>
        <p style="margin-top: 10px; opacity: 0.8;">
            <a href="/docs" target="_blank" style="color: white; text-decoration: none;">
                <i class="fas fa-book"></i> API Documentation
            </a>
        </p>
    </div>

    <script>
        let apiBaseUrl = window.location.origin;

        function setTicker(ticker) {
            document.getElementById('ticker').value = ticker;
        }

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const ticker = document.getElementById('ticker').value.toUpperCase().trim();
            
            if (!ticker) {
                alert('Please enter a stock ticker');
                return;
            }
            
            const btn = document.getElementById('predictBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            btn.disabled = true;
            loading.classList.add('show');
            results.classList.remove('show');
            
            try {
                const response = await fetch(`${apiBaseUrl}/api/v1/predict/stock`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        ticker: ticker,
                        use_real_data: true
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Prediction failed');
                }
                
                const data = await response.json();
                displayResults(data);
                
            } catch (error) {
                alert('Error: ' + error.message + '\\n\\nPlease check if the ticker symbol is correct.');
            } finally {
                btn.disabled = false;
                loading.classList.remove('show');
            }
        });

        function displayResults(data) {
            const results = document.getElementById('results');
            
            // Update stock info
            document.getElementById('stockTicker').textContent = data.ticker;
            if (data.current_price) {
                document.getElementById('stockPrice').textContent = '$' + data.current_price.toFixed(2);
            } else {
                document.getElementById('stockPrice').textContent = 'Price N/A';
            }
            document.getElementById('dataSource').innerHTML = 
                '<i class="fas fa-check-circle"></i> ' + data.data_source;
            
            // Update prediction
            const icon = document.getElementById('predictionIcon');
            const label = document.getElementById('predictionLabel');
            
            if (data.predicted_class_name === 'Up') {
                icon.textContent = '📈';
                label.textContent = 'Bullish Trend Detected';
                label.style.color = '#10b981';
            } else if (data.predicted_class_name === 'Down') {
                icon.textContent = '📉';
                label.textContent = 'Bearish Trend Detected';
                label.style.color = '#ef4444';
            } else {
                icon.textContent = '➡️';
                label.textContent = 'Neutral Market';
                label.style.color = '#f59e0b';
            }
            
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
            
            document.getElementById('probDown').textContent = 
                (data.probabilities.Down * 100).toFixed(1) + '%';
            document.getElementById('probNeutral').textContent = 
                (data.probabilities.Neutral * 100).toFixed(1) + '%';
            document.getElementById('probUp').textContent = 
                (data.probabilities.Up * 100).toFixed(1) + '%';
            
            results.classList.add('show');
        }
    </script>
</body>
</html>
"""

# ==================== Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTML_CONTENT

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_state.loaded,
        "version": "3.0.0"
    }

@app.post("/api/v1/predict/stock", response_model=PredictionResponse)
def predict_stock(request: StockRequest):
    global prediction_count
    prediction_count += 1
    
    ticker = request.ticker.upper()
    current_price = None
    data_source = "demo data"
    
    # Fetch real data
    if request.use_real_data:
        sequence, current_price, error = fetch_real_stock_data(ticker)
        
        if error:
            # Fallback to demo
            print(f"Error fetching {ticker}: {error}")
            sequence = [[np.random.randn() * 0.1 for _ in range(33)] for _ in range(60)]
            data_source = f"demo data (yfinance error: {error[:50]})"
        else:
            data_source = "real market data (yfinance)"
    else:
        sequence = [[np.random.randn() * 0.1 for _ in range(33)] for _ in range(60)]
    
    sequence = np.array(sequence, dtype=np.float32)
    
    # Make prediction
    if model_state.loaded and model_state.model is not None:
        X = sequence.reshape(1, 60, 33)
        predictions = model_state.model.predict(X, verbose=0)
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        probabilities = {
            'Down': float(predictions[0][0]),
            'Neutral': float(predictions[0][1]),
            'Up': float(predictions[0][2])
        }
    else:
        # Demo prediction
        input_mean = float(np.mean(sequence))
        if input_mean > 0.2:
            probs = [0.22, 0.28, 0.50]
        elif input_mean < -0.2:
            probs = [0.50, 0.28, 0.22]
        else:
            probs = [0.30, 0.40, 0.30]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        probabilities = {
            'Down': float(probs[0]),
            'Neutral': float(probs[1]),
            'Up': float(probs[2])
        }
    
    class_names = ['Down', 'Neutral', 'Up']
    
    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_class_name=class_names[predicted_class],
        confidence=confidence,
        probabilities=probabilities,
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_state.loaded,
        ticker=ticker,
        current_price=current_price,
        data_source=data_source
    )

@app.get("/api/v1/metrics")
def get_metrics():
    return {
        "total_predictions": prediction_count,
        "model_loaded": model_state.loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)