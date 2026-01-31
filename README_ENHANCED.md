# 🚀 Advanced Stock Predictor

Professional-grade stock market prediction system powered by AI with comprehensive technical analysis, risk assessment, and real-time market data.

## ✨ Key Features

### 🤖 AI-Powered Predictions
- **Bi-LSTM + Attention Neural Network** for accurate market predictions
- Real-time analysis with confidence scores
- Three-class prediction: Bullish, Bearish, Neutral

### 📊 Technical Analysis
- **13+ Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD & Signal Line
  - Bollinger Bands (Upper/Lower)
  - SMA (20-day & 50-day)
  - EMA (12-day & 26-day)
  - Volume Analysis
  - Volatility Metrics

### 🛡️ Risk Management
- Confidence Level Assessment (High/Medium/Low)
- Volatility Rating
- Trend Strength Analysis (0-100 scale)
- Support & Resistance Levels
- Risk/Reward Ratio Calculation

### 🎯 Price Targets
- 1-Day Price Target
- 1-Week Price Target
- Intelligent Stop-Loss Suggestions

### 📈 Market Context
- Price Momentum Analysis
- Trading Volume Status
- Market Sentiment Detection
- Sector Trend Identification (when available)

### 💼 Trading Recommendations
- Smart buy/sell signals: **Strong Buy, Buy, Hold, Sell, Strong Sell**
- Based on multiple factors:
  - Prediction confidence
  - RSI levels
  - Volatility conditions
  - Risk metrics

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│  ┌───────────────────────────────────┐  │
│  │   Stock Data Fetcher (yfinance)   │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │   Technical Indicators Engine     │  │
│  │   (RSI, MACD, BB, SMA, EMA, etc)  │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │   Bi-LSTM + Attention Model       │  │
│  │   (60 timesteps, 33 features)     │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │   Risk Assessment Engine          │  │
│  │   (Volatility, Support/Resistance)│  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │   Recommendation Generator        │  │
│  │   (Smart Buy/Sell Signals)        │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/riya-tech02/stock-predictor-pro.git
cd stock-predictor-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app_enhanced.py
```

4. **Access the application**
Open your browser and navigate to:
```
http://localhost:8000
```

## 🚀 Deployment

### Deploy to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app_enhanced:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: 3.10 or higher

### Deploy to Heroku

```bash
heroku create your-app-name
git push heroku main
```

Add a `Procfile`:
```
web: uvicorn app_enhanced:app --host 0.0.0.0 --port $PORT
```

## 📡 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_load_time": "2024-01-30T12:00:00",
  "cache_size": 5,
  "timestamp": "2024-01-30T12:05:00"
}
```

#### 2. Predict Stock
```http
POST /api/v1/predict/stock
Content-Type: application/json

{
  "ticker": "AAPL",
  "use_real_data": true
}
```

Response:
```json
{
  "predicted_class": 2,
  "predicted_class_name": "Up",
  "confidence": 0.752,
  "probabilities": {
    "Down": 0.089,
    "Neutral": 0.159,
    "Up": 0.752
  },
  "ticker": "AAPL",
  "current_price": 178.50,
  "change_percent": 1.23,
  "change_amount": 2.17,
  "technical_indicators": {
    "rsi": 58.3,
    "macd": 2.14,
    "macd_signal": 1.87,
    "bb_upper": 182.45,
    "bb_lower": 174.32,
    "sma_20": 176.89,
    "sma_50": 172.15,
    "volume_trend": "High",
    "volatility": 24.5
  },
  "risk_metrics": {
    "confidence_level": "High",
    "volatility_rating": "Medium",
    "trend_strength": 68.5,
    "support_level": 174.12,
    "resistance_level": 182.78,
    "risk_reward_ratio": 1.85
  },
  "market_context": {
    "price_momentum": "Bullish",
    "trading_volume_status": "Above Average",
    "market_sentiment": "Bullish"
  },
  "target_price_1d": 180.20,
  "target_price_1w": 185.50,
  "stop_loss_suggestion": 173.15,
  "recommendation": "Strong Buy",
  "data_quality": "Excellent",
  "data_source": "Live market data",
  "timestamp": "2024-01-30T12:05:00"
}
```

#### 3. Get Prediction History
```http
GET /api/v1/history/{ticker}?limit=20
```

#### 4. Batch Predictions
```http
POST /api/v1/batch-predict
Content-Type: application/json

["AAPL", "TSLA", "GOOGL"]
```

#### 5. System Statistics
```http
GET /api/v1/stats
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:
```env
# API Configuration
PORT=8000
HOST=0.0.0.0

# Cache Settings
CACHE_DURATION=300  # 5 minutes

# Rate Limiting
RATE_LIMIT_WINDOW=60  # 1 minute
MAX_REQUESTS_PER_WINDOW=10

# Model Settings
MODEL_PATH=models/production_model.h5
```

### Cache Management
- **Cache Duration**: 5 minutes (configurable)
- **Cache Strategy**: LRU (Least Recently Used)
- **Automatic Invalidation**: Based on timestamp

### Rate Limiting
- **Default**: 10 requests per minute per client
- **Configurable**: Adjust in environment variables
- **Protection**: Prevents API abuse and rate limit hits

## 📊 Supported Stocks

The system works with any valid stock ticker available on Yahoo Finance, including:

- **US Stocks**: AAPL, TSLA, GOOGL, MSFT, AMZN, META, NVDA, etc.
- **ETFs**: SPY, QQQ, IWM, etc.
- **International**: Add exchange suffix (e.g., `RELIANCE.NS` for NSE)

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live market data integration
- **Interactive Charts**: Visual representation of predictions
- **Professional UI**: Modern, clean interface
- **Fast Performance**: Optimized for quick analysis

## 🔒 Security Features

- **Rate Limiting**: Prevents abuse
- **Input Validation**: Sanitizes user inputs
- **Error Handling**: Graceful error management
- **CORS Protection**: Configurable origins
- **Data Caching**: Reduces API calls and improves speed

## 📈 Performance Optimizations

1. **Smart Caching**: 5-minute cache for market data
2. **Rate Limit Protection**: Prevents Yahoo Finance rate limits
3. **Batch Processing**: Support for multiple ticker analysis
4. **Lazy Loading**: Model loads on startup
5. **Efficient Calculations**: Vectorized numpy operations

## 🛠️ Development

### Project Structure
```
stock-predictor-pro/
├── app.py                  # Original application
├── app_enhanced.py         # Enhanced version with all features
├── index_enhanced.html     # Professional frontend
├── requirements.txt        # Dependencies
├── models/
│   └── production_model.h5 # Trained model (if available)
├── templates/
│   └── index.html         # Frontend template
└── README.md              # This file
```

### Running in Development Mode

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Install pytest
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   - Solution: The app has built-in caching and delay mechanisms
   - Uses cached data when API is rate-limited

2. **Model Not Loading**
   - Solution: System falls back to demo mode with smart heuristics
   - Place model file in `models/production_model.h5`

3. **Yahoo Finance Connection Issues**
   - Solution: Check internet connection
   - System provides estimated data as fallback

## 📝 TODO / Future Enhancements

- [ ] Add historical prediction accuracy tracking
- [ ] Implement news sentiment analysis
- [ ] Add comparison between multiple stocks
- [ ] Real-time WebSocket updates
- [ ] Export reports as PDF
- [ ] Mobile app version
- [ ] Multi-timeframe analysis (1min, 5min, 1hour, daily)
- [ ] Portfolio tracking and analysis
- [ ] Options trading suggestions
- [ ] Integration with broker APIs

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Disclaimer**: This tool is for educational and informational purposes only. It should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.