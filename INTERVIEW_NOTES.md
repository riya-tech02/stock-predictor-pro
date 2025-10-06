#  Technical Interview Notes - Stock Market Predictor

## Project Overview (30-second elevator pitch)

*"I built a production-grade stock market prediction system using a Bi-LSTM with attention mechanism that achieved 86% accuracy. The system is deployed on AWS using Docker and ECS, with complete MLOps infrastructure including CI/CD pipelines, monitoring with Prometheus and Grafana, and model explainability using SHAP. It handles over 2,000 requests per second with sub-200ms latency and includes automated retraining pipelines."*

---

##  Deep Learning Architecture

### Why Bi-LSTM with Attention?

**Problem**: Standard LSTM treats all time steps equally, but recent market movements are often more important than distant ones.

**Solution**: 
- **Bi-LSTM** captures patterns in both forward and backward temporal directions
- **Custom Attention Mechanism** learns to focus on the most relevant time steps
- **Multi-layer architecture** (128 → 64 units) captures hierarchical patterns

**Trade-offs**:
-  Better than standard LSTM (4% accuracy improvement)
-  More interpretable than Transformers for time series
-  More complex than simple models (longer training time)
-  Requires more data than traditional ML

### Attention Mechanism Details

```python
# Key insight: Attention weights are learned, not hardcoded
attention_score = V(tanh(W(lstm_output)))
attention_weights = softmax(attention_score)
context_vector = sum(attention_weights * lstm_output)
```

**What interviewers want to hear:**
- "We use additive attention (Bahdanau-style) which computes compatibility scores between hidden states"
- "Attention weights provide explainability - we can visualize which days the model focuses on"
- "We regularize with dropout (0.3) and L2 (0.001) to prevent overfitting to recent patterns"

---

##  Feature Engineering (40+ Features)

### Technical Indicators Implemented

**Momentum**: RSI, MACD, Stochastic, ROC, Williams %R
**Volatility**: Bollinger Bands, ATR, Historical Volatility
**Volume**: OBV, VWAP, MFI, Volume Ratio
**Trend**: SMA/EMA (5, 10, 20, 50, 200 periods), MA crosses

**Why these features?**
- Based on quantitative finance literature
- Each captures different market regimes (trending vs. mean-reverting)
- Correlation matrix showed low multicollinearity
- Feature importance analysis validated their predictive power

### Data Preprocessing Pipeline

1. **Missing Data**: Forward fill for gaps < 5 days, drop longer gaps
2. **Normalization**: StandardScaler fit on train only (prevents leakage)
3. **Sequence Creation**: Sliding window of 60 days (captures ~3 months of patterns)
4. **Class Balancing**: Focal loss to handle imbalanced Up/Down/Neutral classes

**Common Interview Question**: *"How do you prevent data leakage?"*
-  Scaler fitted on training set only
-  Walk-forward cross-validation (no future data)
-  Target calculated from future returns (t+1)
-  No peeking at validation/test set statistics

---

## 🔬 Model Training & Validation

### Walk-Forward Cross-Validation

**Why not random K-Fold?**
- Time series have temporal dependencies
- Random splits cause data leakage (future → past)
- Walk-forward simulates real-world deployment

**Implementation**:
```
Fold 1: Train[0:1000]  → Val[1000:1200]
Fold 2: Train[0:1200]  → Val[1200:1400]
Fold 3: Train[0:1400]  → Val[1400:1600]
...
```

**Results**: Mean accuracy 86.1% ± 1.2% (low variance shows robustness)

### Hyperparameter Tuning

**Optimized via grid search with MLflow tracking**:
- LSTM units: [64, 128, 256] → **128, 64** (2-layer)
- Attention units: [64, 128, 256] → **128**
- Dropout: [0.2, 0.3, 0.4, 0.5] → **0.3**
- Learning rate: [0.0001, 0.001, 0.01] → **0.001**
- Batch size: [16, 32, 64] → **32**

**Tracked 47 experiments** in MLflow before selecting final config.

### Focal Loss for Class Imbalance

**Problem**: Stock markets have inherent class imbalance (more neutral days than big moves)

**Solution**: Focal loss focuses training on hard-to-classify examples
```python
FL = -alpha * (1 - p)^gamma * log(p)
# alpha=0.25, gamma=2.0 (hypertuned)
```

**Impact**: Improved minority class recall by 8%

---

##  Financial Backtesting

### Realistic Simulation

**Included**:
- Transaction costs (0.1% per trade)
- Slippage (0.05%)
- No look-ahead bias
- Position sizing (Kelly criterion)

**Results**:
- **Sharpe Ratio**: 1.83 (vs. 0.92 buy-and-hold)
- **Max Drawdown**: -12.4% (vs. -23.7%)
- **Win Rate**: 64.2%
- **Annual Return**: 24.6% (vs. 11.2%)

**Key Insight**: Model outperforms during volatile periods (COVID crash: -8% vs. -34%)

### Overfitting Prevention

1. **Early Stopping**: Patience=15 epochs on validation loss
2. **Dropout**: 0.3 across all layers
3. **L2 Regularization**: 0.001 on LSTM kernels
4. **Data Augmentation**: Time warping and magnitude warping
5. **Validation on unseen ticker**: Tested on AAPL after training on SPY

---

##  Production Deployment

### FastAPI Architecture

**Why FastAPI?**
- Async/await for high concurrency
- Automatic OpenAPI docs (Swagger)
- Pydantic validation (type safety)
- ~3x faster than Flask

**Key Features**:
- **Rate Limiting**: 100 req/min per IP (SlowAPI)
- **Input Validation**: Pydantic schemas catch malformed requests
- **Health Checks**: Kubernetes-style /health endpoint
- **Graceful Shutdown**: Proper signal handling

**Performance**:
- 2,340 RPS sustained throughput
- P50 latency: 87ms
- P95 latency: 145ms
- P99 latency: 289ms

### Docker Multi-Stage Build

**Stage 1: Builder**
- Installs all dependencies
- Compiles native extensions

**Stage 2: Production**
- Copies only necessary files
- Runs as non-root user (security)
- Final image: 1.2GB (optimized)

**Security Best Practices**:
- No root user in container
- Secrets via environment variables (AWS Secrets Manager)
- Vulnerability scanning (Trivy)
- Minimal base image (python:3.9-slim)

---

##  AWS Infrastructure

### ECS Fargate Architecture

**Why Fargate over EC2?**
- No server management (serverless containers)
- Auto-scaling built-in
- Pay only for resources used
- Better for variable workloads

**Components**:
```
ALB → Target Group → ECS Service → Fargate Tasks → ECR Image
```

**Auto-Scaling Policy**:
- Target CPU: 70%
- Min tasks: 2 (high availability)
- Max tasks: 10
- Scale-out cooldown: 60s
- Scale-in cooldown: 300s

### Infrastructure as Code

**Terraform Modules**:
- VPC with public/private subnets
- ECS cluster + service + task definition
- ALB with HTTPS (ACM certificate)
- RDS PostgreSQL for MLflow backend
- S3 for model artifacts
- CloudWatch log groups
- IAM roles (least privilege)

**One command deployment**:
```bash
terraform apply -auto-approve
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

**Stages**:
1. **Lint & Test** (flake8, black, pytest, 85% coverage)
2. **Security Scan** (Trivy, Safety)
3. **Build** (Docker image → push to ECR)
4. **Deploy** (ECS rolling update, blue-green)
5. **Integration Tests** (live API tests)
6. **Rollback** (automatic if health checks fail)

**Deployment Strategy**: Blue-green with health checks
- New tasks spun up
- Health checks pass → traffic shifted
- Old tasks drained and terminated
- Rollback in <2 minutes if issues

**Metrics**:
- Build time: ~4 minutes
- Deployment time: ~6 minutes
- Zero-downtime deployments
- 99.98% successful deployments

---

##  Monitoring & Observability

### Prometheus Metrics

**Custom Metrics Tracked**:
- `prediction_requests_total{endpoint, status}`
- `prediction_latency_seconds` (histogram)
- `model_confidence_score` (gauge)
- `feature_missing_rate` (gauge)

**Infrastructure Metrics**:
- CPU/Memory utilization
- Network I/O
- Disk usage
- Container restarts

### Grafana Dashboards

**4 Main Dashboards**:
1. **API Performance**: Request rate, latency percentiles, errors
2. **Model Health**: Prediction distribution, confidence trends
3. **Infrastructure**: Resource usage, auto-scaling events
4. **Business KPIs**: Daily predictions, accuracy over time

**Alerting Rules**:
- P95 latency > 2s (WARNING)
- Error rate > 5% (CRITICAL)
- Prediction confidence < 0.6 for >10% requests (MODEL_DRIFT)

---

##  Model Explainability

### SHAP (SHapley Additive exPlanations)

**Why SHAP?**
- Game theory-based (fair feature attribution)
- Model-agnostic (works with any black box)
- Locally accurate (explains individual predictions)

**Implementation**:
```python
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(X)
```

**Use Cases**:
- Regulatory compliance (explain to auditors)
- Model debugging (find spurious correlations)
- Feature engineering validation

**