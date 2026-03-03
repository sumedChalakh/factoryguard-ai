# FactoryGuard AI 🏭⚡

> **Unplanned factory equipment failures cost manufacturers billions annually.**
> FactoryGuard AI is a production-ready, ML-powered predictive maintenance API that scores equipment failure risk in real-time from sensor data — with built-in explainability so engineers know *why* a machine is flagged, not just *that* it is.

[![CI](https://github.com/sumedChalakh/factoryguard-ai/actions/workflows/test.yml/badge.svg)](https://github.com/sumedChalakh/factoryguard-ai/actions)
![Python](https://img.shields.io/badge/python-3.10%20|%203.12-blue)
![Flask](https://img.shields.io/badge/framework-Flask-lightgrey)


---

## 📌 What This Project Does

FactoryGuard AI accepts three real-time sensor readings from industrial equipment:

| Input | Description |
|---|---|
| `temperature` | Equipment operating temperature (°C) |
| `vibration` | Vibration intensity (mm/s) |
| `pressure` | Operating pressure (bar) |

And returns:

- **Failure probability score** → `/predict`
- **SHAP-based feature-level explanation** → `/explain` *(why is this machine at risk?)*

### Why SHAP Explainability?
Knowing a machine has a 95% failure probability isn't enough for a maintenance engineer. FactoryGuard's `/explain` endpoint returns SHAP values that show *which sensor reading* is driving the risk — enabling targeted, faster intervention.

---

## 🚀 Production Features

| Feature | Detail |
|---|---|
| 🔐 API Key Auth | Header or Bearer token protection on inference routes |
| 🚦 Rate Limiting | Configurable per-endpoint limits (Redis-ready for distributed deployments) |
| 🌐 CORS | Configurable allowlist |
| 📊 Prometheus Metrics | `/metrics` endpoint for scraping |
| 🩺 Health Probes | Liveness + Readiness probes for Kubernetes |
| 📄 OpenAPI / Swagger | Auto-generated API docs at `/docs` and `/swagger` |
| 🔍 SHAP Explainability | Feature-level failure driver breakdown |
| 🧪 Automated Tests | 9 pytest cases covering all critical paths |
| ⚙️ CI/CD | GitHub Actions for Python 3.10 & 3.12 |
| 🐳 Docker | Container-ready with Dockerfile |
| 📝 Structured Logging | Request IDs and structured log output |

---

## ⚡ Performance Benchmarks

Benchmarked locally using `src/latency_benchmark.py` under single-user conditions:

### `/predict` endpoint — 50 requests, no concurrency

| Metric | Result |
|---|---|
| Success Rate | 50/50 (0% error) |
| Average Latency | **18.88 ms** |
| p50 Latency | 16.74 ms |
| p95 Latency | 32.27 ms |
| p99 Latency | 32.97 ms |
| Max Latency | 33.25 ms |

### `/explain` endpoint — 20 requests, no concurrency

| Metric | Result |
|---|---|
| Average Latency | **20.25 ms** |
| p95 Latency | 34.32 ms |
| Error Rate | 0% |

✅ Both endpoints comfortably meet the <50ms real-time inference requirement.

---

## 🗂 Project Structure

```
factoryguard-ai/
├── api/                  # Flask app, route handlers, schemas, OpenAPI spec, docs UI
├── src/                  # Model training, evaluation, feature engineering, benchmark utils
├── notebooks/            # EDA and model development notebooks
├── docker/               # Dockerfile and container configs
├── docs/                 # Design notes and API documentation
├── .github/workflows/    # GitHub Actions CI pipeline
├── test_api.py           # Full API test suite (9 tests)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md
```

---

## 🛠 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sumedChalakh/factoryguard-ai.git
cd factoryguard-ai
```

### 2. Set up a virtual environment

```bash
# Option A: Conda
conda create -n factoryguard python=3.12
conda activate factoryguard

# Option B: venv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API key and settings
```

### 5. Run tests

```bash
pytest test_api.py -v
```
### Or

```bash
pytest -v
```

### 6. Start the API

```bash
cd factoryguard-ai
python -m api.app
```

API will be live at: `http://localhost:5000`
Swagger docs at: `http://localhost:5000/docs`

---

## 🔌 API Reference

### Health Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Basic service status |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks model/scaler) |

### Inference Endpoints *(API key required)*

#### `POST /predict` — Failure Risk Score

**Request:**
```json
{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

**Response:**
```json
{
  "failure_probability": 0.95
}
```

---

#### `POST /explain` — Failure Risk + SHAP Explanation

**Request:**
```json
{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

**Response:**
```json
{
  "failure_probability": 0.95,
  "base_value": 0.5,
  "shap_values": {
    "temperature": 0.12,
    "vibration": 0.38,
    "pressure": 0.05
  }
}
```
> In this example, `vibration` is the dominant driver of failure risk.

---

### Observability & Docs

| Method | Route | Description |
|---|---|---|
| GET | `/openapi.json` | Raw OpenAPI spec |
| GET | `/docs` | Swagger UI |
| GET | `/swagger` | Alternate Swagger UI |
| GET | `/metrics` | Prometheus metrics |

---

## 🔐 Authentication

When `API_KEY` is set in `.env`, protected routes require it via:

```bash
# Header
curl -H "X-API-Key: your-key" http://localhost:5000/predict ...

# Bearer token
curl -H "Authorization: Bearer your-key" http://localhost:5000/predict ...
```

Protected routes: `/predict`, `/explain`

---

## ⚙️ Environment Configuration

See `.env.example` for all options. Key variables:

```env
# App
FLASK_ENV=production
FLASK_DEBUG=0
LOG_LEVEL=INFO

# Security
API_KEY=change-me
API_KEY_HEADER=X-API-Key
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Rate Limiting
DEFAULT_RATE_LIMIT=120 per minute
PREDICT_RATE_LIMIT=60 per minute
EXPLAIN_RATE_LIMIT=30 per minute
RATE_LIMIT_STORAGE_URI=memory://   # Use redis:// for distributed deployments

# Models
MODEL_PATH=models/model.joblib
SCALER_PATH=models/preprocessor.joblib

# Server
HOST=0.0.0.0
PORT=5000
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -f docker/Dockerfile -t factoryguard:latest .

# Run
docker run -p 5000:5000 --env-file .env factoryguard:latest
```

---

## ☸️ Kubernetes Readiness

FactoryGuard is probe-ready out of the box:

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

**Recommended production infra:**
- TLS termination at ingress/reverse proxy
- Redis-backed rate limiter (`RATE_LIMIT_STORAGE_URI=redis://...`)
- Centralized logging (ELK / CloudWatch)
- Metrics scraping + alerting (Prometheus + Grafana)

---

## 🧪 Testing

```bash
pytest test_api.py -v
```

**Test coverage (9 tests):**
- ✅ Health / liveness / readiness endpoints
- ✅ Prediction success path
- ✅ Explanation success path
- ✅ Schema validation errors (bad input)
- ✅ API key enforcement
- ✅ Docs and metrics endpoints

---

## 🔁 CI/CD

GitHub Actions workflow: `.github/workflows/test.yml`

Triggers on push/PR to `main` and `develop`. Validates:
- Dependency installation
- pytest execution (Python 3.10 & 3.12)
- Schema JSON loading

---

## 🗺 Roadmap

- [ ] Add training notebook with model evaluation metrics (Accuracy, F1, ROC-AUC)
- [ ] Document dataset source and feature engineering steps
- [ ] Add live demo deployment (Render / Railway)
- [ ] Expand to 5+ sensor features
- [ ] Add batch prediction endpoint `/predict/batch`
- [ ] Integrate alerting webhook for high-risk predictions

---

*Last updated: February 27, 2026 | Internship handoff ready · Deployment ready · CI validated*
