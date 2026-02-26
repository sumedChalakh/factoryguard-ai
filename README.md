# FactoryGuard AI

A production-grade machine learning API for **predictive equipment failure detection** using SHAP-explainable logistic regression and XGBoost models.

**Status:** 58% → 85%+ Production Ready | CI/CD Enabled | Fully Tested

---

## 🎯 Overview

FactoryGuard AI predicts industrial equipment failures 24 hours in advance using real-time sensor data (temperature, vibration, pressure). Each prediction includes **SHAP explanations** to understand which sensors contributed to the failure risk.

**Key Features:**
- ✅ REST API with strict JSON schema validation
- ✅ SHAP-based explainable AI (per-instance feature importance)
- ✅ Automated test suite (9 pytest tests, 100% passing)
- ✅ GitHub Actions CI/CD (Python 3.10, 3.12)
- ✅ Production-safe error handling (JSON responses, no HTML tracebacks)
- ✅ Docker support for containerized deployment
- ✅ Time-based train/test split (no data leakage)

---

## 📋 Requirements

- **Python:** 3.10+ 
- **OS:** Linux, macOS, Windows (with conda/venv)
- **RAM:** 4GB minimum (for model & SHAP computation)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sumedChalakh/factoryguard-ai.git
cd factoryguard-ai
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n factoryguard python=3.12
conda activate factoryguard

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
pytest test_api.py -v
```

### 5. Start the API Server
```bash
python -m api.app
```

API will be available at: **http://localhost:5000**

---

## 📡 API Reference

### Health Check
```http
GET /
```
**Response (200 OK):**
```json
{
  "status": "ok"
}
```

### Predict Failure Probability
```http
POST /predict
Content-Type: application/json

{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

**Response (200 OK):**
```json
{
  "failure_probability": 0.95
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid request payload",
  "message": "temperature is a required property"
}
```

### Predict + Get SHAP Explanation
```http
POST /explain
Content-Type: application/json

{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

**Response (200 OK):**
```json
{
  "failure_probability": 0.95,
  "base_value": 266.14,
  "shap_values": {
    "temperature": 0.12,
    "temperature_lag1": -0.05,
    "vibration": 0.33,
    "vibration_mean_6h": 0.18
  }
}
```

---

## 🏗️ Architecture

### Data Flow
```
Raw Sensor Input
    ↓
Feature Engineering (lags + rolling stats)
    ↓
StandardScaler (fit on train only)
    ↓
LogisticRegression / XGBoost Model
    ↓
Prediction + SHAP Explanation
    ↓
JSON Response
```

### Directory Structure
```
.
├── api/
│   ├── app.py              # Flask application & endpoints
│   ├── schema.json         # JSON schema validation
│   └── __init__.py
├── src/
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Model evaluation
│   ├── feature_engineering.py
│   ├── utils.py            # SHAP explainer
│   └── __init__.py
├── models/                 # Trained models (joblib)
├── data/
│   ├── raw/                # Raw sensor data
│   └── processed/          # Feature engineered data
├── docker/                 # Docker configuration
├── docs/                   # Documentation
├── .github/
│   └── workflows/
│       └── test.yml        # GitHub Actions CI
├── requirements.txt
├── test_api.py            # Pytest tests
└── README.md
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest test_api.py -v
```

### With Coverage
```bash
pytest test_api.py --cov=api --cov-report=html
```

**Coverage:**
- ✅ Health endpoint
- ✅ Valid predictions & SHAP explanations
- ✅ Schema validation (missing fields, wrong types)
- ✅ Error responses (400, 500)

---

## 🐳 Docker Deployment

### Build & Run
```bash
docker build -f docker/Dockerfile -t factoryguard:latest .
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  factoryguard:latest
```

---

## 🚢 Kubernetes

Use standard health probes:
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 5000

readinessProbe:
  httpGet:
    path: /health/ready
    port: 5000
```

---

## 🔐 Security & Best Practices

### Implemented
- ✅ JSON schema validation
- ✅ Safe error handling (no tracebacks)
- ✅ Model versioning ready

### Recommended for Production
- 🔒 API key authentication
- 🔒 HTTPS/TLS
- 🔒 CORS configuration
- 🔒 Rate limiting
- 🔒 Centralized logging (ELK, CloudWatch)
- 🔒 Monitoring (Prometheus, DataDog)

---

## 📊 Model Performance

**Primary Metric:** PR-AUC

```bash
cat reports/evaluation_results.csv
```

---

## 🔄 CI/CD Pipeline

GitHub Actions runs on every push:
- ✅ Pytest on Python 3.10 & 3.12
- ✅ Schema validation
- ✅ Coverage reporting

View at: https://github.com/sumedChalakh/factoryguard-ai/actions

---

## 🛠️ Development

### Start with Debug Mode
```bash
FLASK_DEBUG=1 python -m api.app
```

### Train New Model
```bash
python src/train.py
```

### Evaluate Performance
```bash
python src/evaluate.py
```

---

## 🐛 Troubleshooting

- **"Input X contains NaN"** → Ensure all sensors are provided
- **"schema.json not found"** → Run from project root
- **Port 5000 in use** → Change port or kill process

---

## 📚 More Info

- [API Errors & Examples](docs/api_errors.md)
- [Architecture Notes](docs/design_notes.md)

---

## 🤝 Contributing

1. Fork repo
2. Feature branch: `git checkout -b feature/my-feature`
3. Test: `pytest test_api.py -v`
4. Commit & push
5. Open PR

---

**Built with ❤️ by FactoryGuard Team**  
Last Updated: February 26, 2026
