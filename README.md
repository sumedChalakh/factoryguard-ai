# FactoryGuard AI

FactoryGuard AI is an ML-powered predictive maintenance API for equipment failure risk scoring and feature-level explainability.

**Final Project Status:** Internship handoff ready, deployment ready for project scope, CI validated.

## 1) What This Project Does

FactoryGuard AI accepts three sensor inputs:
- `temperature`
- `vibration`
- `pressure`

It returns:
- a failure probability (`/predict`)
- optional SHAP explanation (`/explain`)

This repository includes:
- model-serving API
- schema validation
- security controls (API key, CORS, rate limiting)
- observability (health/readiness/metrics)
- tests + CI pipeline

## 2) Production Features Implemented

- Flask API with strict JSON schema validation
- Model-serving endpoints (`/predict`, `/explain`)
- Health and readiness probes (`/`, `/health/live`, `/health/ready`)
- Prometheus-compatible metrics endpoint (`/metrics`)
- OpenAPI + docs endpoints (`/openapi.json`, `/docs`, `/swagger`)
- API key protection for inference routes
- Configurable CORS allowlist
- Configurable rate limiting per endpoint
- Structured logging and request IDs
- Automated tests (9 pytest tests)
- GitHub Actions CI for Python 3.10 and 3.12

## 3) Quick Start

### 3.1 Clone Repository

```bash
git clone https://github.com/sumedChalakh/factoryguard-ai.git
cd factoryguard-ai
```

### 3.2 Create Virtual Environment

```bash
# Option A: Conda
conda create -n factoryguard python=3.12
conda activate factoryguard

# Option B: venv
python -m venv venv
# Windows
venv\Scripts\activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Configure Environment

```bash
copy .env.example .env
```

Update `.env` values for your machine/security.

### 3.5 Run Tests

```bash
pytest test_api.py -v
```

### 3.6 Run API

```bash
python -m api.app
```

Default base URL: `http://localhost:5000`

## 4) Environment Configuration

Key environment variables (see `.env.example`):

```bash
FLASK_ENV=production
FLASK_DEBUG=0
LOG_LEVEL=INFO

API_KEY=change-me
API_KEY_HEADER=X-API-Key

ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

DEFAULT_RATE_LIMIT=120 per minute
PREDICT_RATE_LIMIT=60 per minute
EXPLAIN_RATE_LIMIT=30 per minute
RATE_LIMIT_STORAGE_URI=memory://

MODEL_PATH=models/model.joblib
SCALER_PATH=models/preprocessor.joblib

HOST=0.0.0.0
PORT=5000
```

### Notes
- If `API_KEY` is set, `/predict` and `/explain` require it.
- For distributed/real production, set `RATE_LIMIT_STORAGE_URI` to Redis.

## 5) API Contract

### 5.1 Health

#### `GET /`
Basic service status.

#### `GET /health/live`
Liveness probe.

#### `GET /health/ready`
Readiness probe (model/scaler readiness).

### 5.2 Prediction

#### `POST /predict`

Request:
```json
{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

Response:
```json
{
  "failure_probability": 0.95
}
```

#### `POST /explain`

Request:
```json
{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

Response shape:
```json
{
  "failure_probability": 0.95,
  "base_value": 0.5,
  "shap_values": {
    "temperature": 0.0,
    "vibration": 0.0,
    "pressure": 0.0
  }
}
```

### 5.3 Docs & Metrics

- `GET /openapi.json`
- `GET /docs`
- `GET /swagger`
- `GET /metrics`

## 6) Authentication

When `API_KEY` is configured, pass it as either:
- header: `X-API-Key: <your-key>`
- or bearer token: `Authorization: Bearer <your-key>`

Protected routes:
- `/predict`
- `/explain`

## 7) Testing

Run all tests:

```bash
pytest test_api.py -v
```

Current suite: **9 tests**, covering:
- health/live/ready endpoints
- prediction and explanation success paths
- schema validation errors
- API key enforcement behavior
- docs + metrics endpoints

## 8) CI/CD

Workflow: `.github/workflows/test.yml`

Runs on push/PR (`main`, `develop`) and validates:
- dependency installation
- pytest execution
- schema JSON loading

Historical failed runs may still appear in Actions history; latest run status is the source of truth.

## 9) Deployment

### 9.1 Docker

```bash
docker build -f docker/Dockerfile -t factoryguard:latest .
docker run -p 5000:5000 --env-file .env factoryguard:latest
```

### 9.2 Kubernetes Probes

- Liveness path: `/health/live`
- Readiness path: `/health/ready`

### 9.3 Recommended Infra Controls

- TLS termination at ingress/reverse proxy
- Redis-backed limiter storage
- centralized logging (ELK/CloudWatch)
- metrics scraping + alerting (Prometheus/Grafana)

## 10) Project Structure

```text
api/                 Flask app, schemas, OpenAPI, docs UI
src/                 Training, evaluation, feature engineering, utilities
docker/              Docker files
docs/                Design and API notes
.github/workflows/   CI pipeline
test_api.py          API test suite
requirements.txt     Dependencies
```

## 11) Internship Handoff Notes

- Project is complete for internship demonstration/deployment scope.
- API is secured, observable, and CI-validated.
- Documentation is finalized in this README and OpenAPI docs.

Last updated: February 27, 2026
