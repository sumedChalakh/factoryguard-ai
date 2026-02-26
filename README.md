# FactoryGuard AI

FactoryGuard AI is a production-ready ML API for predictive equipment failure detection using sensor data.

Status: Internship deployment ready (project scope complete) | CI/CD enabled | tests passing.

## Overview

The API predicts failure probability using input sensors:
- `temperature`
- `vibration`
- `pressure`

It also supports SHAP-based explanation output for model interpretability.

## Implemented Features

- REST API with request/response JSON schema validation
- Prediction endpoint: `/predict`
- Explanation endpoint: `/explain`
- Health endpoints: `/`, `/health/live`, `/health/ready`
- Metrics endpoint: `/metrics` (Prometheus format)
- OpenAPI spec and docs: `/openapi.json`, `/docs`, `/swagger`
- API key protection for inference endpoints
- CORS via environment configuration
- Configurable rate limiting
- Structured logging + request ID headers
- CI pipeline on GitHub Actions (Python 3.10 and 3.12)
- Automated pytest suite (9 tests)

## Quick Start

### 1) Clone

```bash
git clone https://github.com/sumedChalakh/factoryguard-ai.git
cd factoryguard-ai
```

### 2) Create environment

```bash
# conda
conda create -n factoryguard python=3.12
conda activate factoryguard

# OR venv
python -m venv venv
# Windows
venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment

```bash
copy .env.example .env
```

### 5) Run tests

```bash
pytest test_api.py -v
```

### 6) Start API

```bash
python -m api.app
```

Base URL: `http://localhost:5000`

## Security Configuration

Set these values in `.env` for secured inference:

```bash
API_KEY=change-me
API_KEY_HEADER=X-API-Key
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
DEFAULT_RATE_LIMIT=120 per minute
PREDICT_RATE_LIMIT=60 per minute
EXPLAIN_RATE_LIMIT=30 per minute
RATE_LIMIT_STORAGE_URI=memory://
```

Notes:
- If `API_KEY` is set, `/predict` and `/explain` require it.
- For multi-instance production, use Redis for `RATE_LIMIT_STORAGE_URI`.

## API Summary

### GET `/`
Basic health check.

### GET `/health/live`
Liveness probe.

### GET `/health/ready`
Readiness probe (returns model readiness).

### POST `/predict`
Returns failure probability.

Request body:
```json
{
  "temperature": 60,
  "vibration": 29,
  "pressure": 102
}
```

### POST `/explain`
Returns failure probability + SHAP contribution map.

### GET `/metrics`
Prometheus metrics.

### Docs
- `/docs`
- `/swagger`
- `/openapi.json`

## Testing

Run:

```bash
pytest test_api.py -v
```

Current test scope includes:
- health/liveness/readiness
- predict/explain happy paths
- schema validation failures
- API key enforcement behavior
- docs and metrics endpoint checks

## CI/CD

GitHub Actions workflow (`.github/workflows/test.yml`) runs on push/PR for `main` and `develop`.

Pipeline checks:
- dependency installation
- pytest execution
- schema JSON validation

## Deployment

### Docker

```bash
docker build -f docker/Dockerfile -t factoryguard:latest .
docker run -p 5000:5000 --env-file .env factoryguard:latest
```

### Kubernetes probes

Use:
- liveness: `/health/live`
- readiness: `/health/ready`

## Project Structure

```text
api/                 Flask app, schema, OpenAPI, docs page
src/                 Training, evaluation, feature engineering, utils
docker/              Docker assets
docs/                Additional docs
.github/workflows/   CI pipeline
test_api.py          API tests
requirements.txt     Dependencies
```

## Notes for Internship Review

- The project is deployment-ready for internship/demo scope.
- Inference endpoints are protected and observable.
- CI is configured and validates every push.
- Previous failed workflow runs remain visible in GitHub history; check the latest run status for current state.

Last updated: February 27, 2026
