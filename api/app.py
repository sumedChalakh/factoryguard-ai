import joblib
import pandas as pd
from flask import Flask, request, jsonify, g
from pathlib import Path
import sys
import os
import json
import logging
from werkzeug.exceptions import HTTPException
from jsonschema import Draft202012Validator
from datetime import datetime, timezone
from hmac import compare_digest
from time import perf_counter
from uuid import uuid4
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import create_features
from src.utils import explain_with_shap

# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
app = Flask(__name__)
DEBUG_MODE = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
app.config["PROPAGATE_EXCEPTIONS"] = DEBUG_MODE

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "120 per minute")
PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "60 per minute")
EXPLAIN_RATE_LIMIT = os.getenv("EXPLAIN_RATE_LIMIT", "30 per minute")
RATE_LIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS.split(",")}})
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri=RATE_LIMIT_STORAGE_URI,
)

REQUEST_COUNTER = Counter(
    "factoryguard_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "factoryguard_request_latency_seconds",
    "Request latency in seconds",
    ["path"],
)

logger.info(f"Flask Debug Mode: {DEBUG_MODE}")
logger.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")

# ------------------------------------------------------------------
# Paths & Model Loading
# ------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model.joblib")))
PREPROCESSOR_PATH = Path(os.getenv("SCALER_PATH", str(BASE_DIR / "models" / "preprocessor.joblib")))
SCHEMA_PATH = BASE_DIR / "api" / "schema.json"

# Model version tracking
MODEL_VERSION = "1.0.0"
MODEL_LOADED_AT = datetime.now(timezone.utc).isoformat()

logger.info(f"Loading model from: {MODEL_PATH}")
logger.info(f"Loading scaler from: {PREPROCESSOR_PATH}")

model = None
scaler = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(PREPROCESSOR_PATH)
    logger.info(f"Model loaded successfully | Version: {MODEL_VERSION}")
except Exception as model_load_exception:
    MODEL_LOAD_ERROR = str(model_load_exception)
    logger.warning(f"Model artifacts not available at startup: {MODEL_LOAD_ERROR}")

with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
    API_SCHEMAS = json.load(schema_file)

logger.info("API schemas loaded successfully")


def _get_validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(API_SCHEMAS[schema_name])


def validate_request_payload(payload: dict, schema_name: str):
    validator = _get_validator(schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        first_error = errors[0]
        return (
            jsonify({
                "error": "Invalid request payload",
                "message": first_error.message
            }),
            400,
        )
    return None


def validate_response_payload(payload: dict, schema_name: str):
    validator = _get_validator(schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        first_error = errors[0]
        raise RuntimeError(f"Response schema validation failed: {first_error.message}")


def is_model_ready() -> bool:
    return model is not None and scaler is not None


def model_unavailable_response():
    return jsonify({
        "error": "Service Unavailable",
        "message": "Model artifacts are not loaded"
    }), 503

# ------------------------------------------------------------------
# Shared preprocessing (CRITICAL)
# ------------------------------------------------------------------
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Convert raw JSON input into model-ready feature vector.
    Matches training pipeline EXACTLY.
    """

    # JSON → DataFrame
    df = pd.DataFrame([data])

    # Required columns for feature engineering
    df["machine_id"] = 0
    df["timestamp"] = pd.Timestamp.now()
    df["failure_24h"] = 0

    # Feature engineering (same as training)
    df_fe = create_features(df)

    # Drop training-only columns
    X = df_fe.drop(columns=["failure_24h", "machine_id", "timestamp"])

    # 🔥 FIX: handle NaNs from lag/rolling windows
    X = X.ffill().fillna(0)

    # Force exact feature order
    X = X[scaler.feature_names_in_]

    return X


def _is_protected_path(path: str) -> bool:
    return path in {"/predict", "/explain"}


def _read_api_key_from_request() -> str:
    header_key = request.headers.get(API_KEY_HEADER, "")
    if header_key:
        return header_key

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()

    return ""


@app.before_request
def before_request_hooks():
    g.request_start = perf_counter()
    g.request_id = request.headers.get("X-Request-Id", str(uuid4()))

    expected_api_key = os.getenv("API_KEY", "").strip()
    if expected_api_key and _is_protected_path(request.path):
        provided_key = _read_api_key_from_request()
        if not provided_key or not compare_digest(provided_key, expected_api_key):
            return jsonify({
                "error": "Unauthorized",
                "message": "Missing or invalid API key"
            }), 401


@app.after_request
def after_request_hooks(response):
    elapsed = perf_counter() - getattr(g, "request_start", perf_counter())
    REQUEST_LATENCY.labels(path=request.path).observe(elapsed)
    REQUEST_COUNTER.labels(
        method=request.method,
        path=request.path,
        status_code=str(response.status_code),
    ).inc()

    response.headers["X-Request-Id"] = getattr(g, "request_id", "unknown")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

# ------------------------------------------------------------------
# Health & Readiness Checks (Kubernetes compatible)
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/health/live", methods=["GET"])
def liveness_probe():
    """Kubernetes liveness probe - is process alive?"""
    return jsonify({
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route("/health/ready", methods=["GET"])
def readiness_probe():
    """Kubernetes readiness probe - can process serve requests?"""
    try:
        if not is_model_ready():
            raise RuntimeError(MODEL_LOAD_ERROR or "Model/scaler not loaded")

        _ = model.predict_proba([[0] * len(scaler.feature_names_in_)])
        
        return jsonify({
            "status": "ready",
            "model_version": MODEL_VERSION,
            "model_loaded_at": MODEL_LOADED_AT,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return jsonify({
            "status": "not_ready",
            "reason": str(e)
        }), 503


# ------------------------------------------------------------------
# Documentation Endpoints
# ------------------------------------------------------------------
@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Serve OpenAPI/Swagger specification"""
    openapi_path = BASE_DIR / "api" / "openapi.json"
    with open(openapi_path, "r") as f:
        return jsonify(json.load(f))


@app.route("/swagger", methods=["GET"])
@app.route("/docs", methods=["GET"])
def swagger_ui():
    """Serve Swagger UI documentation"""
    swagger_path = BASE_DIR / "api" / "swagger.html"
    with open(swagger_path, "r") as f:
        return f.read(), 200, {"Content-Type": "text/html"}


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    logger.warning(f"HTTP Exception: {error.name} | {error.description}")
    return jsonify({
        "error": error.name,
        "message": error.description
    }), error.code


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    if DEBUG_MODE:
        raise error
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500

# ------------------------------------------------------------------
# PREDICT ONLY
# ------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict():
    if not is_model_ready():
        logger.error("Predict requested but model is unavailable")
        return model_unavailable_response()

    data = request.get_json(silent=True)
    if not data:
        logger.warning("Predict called with empty payload")
        return jsonify({"error": "Empty JSON payload"}), 400

    validation_error = validate_request_payload(data, "predict_request")
    if validation_error:
        logger.warning(f"Predict validation failed: {data}")
        return validation_error

    try:
        X = preprocess_input(data)
        X_scaled = scaler.transform(X)
        failure_prob = model.predict_proba(X_scaled)[0][1]

        response_payload = {
            "failure_probability": round(float(failure_prob), 6)
        }

        validate_response_payload(response_payload, "predict_response")
        logger.info(f"Predict succeeded | Probability: {failure_prob:.4f}")
        return jsonify(response_payload)
    except Exception as e:
        logger.error(f"Predict error: {str(e)}", exc_info=True)
        raise

# ------------------------------------------------------------------
# PREDICT + SHAP EXPLANATION
# ------------------------------------------------------------------
@app.route("/explain", methods=["POST"])
@limiter.limit(EXPLAIN_RATE_LIMIT)
def explain():
    if not is_model_ready():
        logger.error("Explain requested but model is unavailable")
        return model_unavailable_response()

    data = request.get_json(silent=True)
    if not data:
        logger.warning("Explain called with empty payload")
        return jsonify({"error": "Empty JSON payload"}), 400

    validation_error = validate_request_payload(data, "explain_request")
    if validation_error:
        logger.warning(f"Explain validation failed: {data}")
        return validation_error

    try:
        X = preprocess_input(data)
        X_scaled = scaler.transform(X)
        failure_prob = model.predict_proba(X_scaled)[0][1]

        explanation = explain_with_shap(
            model,
            pd.DataFrame(X_scaled, columns=X.columns)
        )

        response_payload = {
            "failure_probability": round(float(failure_prob), 6),
            "base_value": explanation["base_value"],
            "shap_values": explanation["shap_values"]
        }

        validate_response_payload(response_payload, "explain_response")
        logger.info(f"Explain succeeded | Probability: {failure_prob:.4f}")
        return jsonify(response_payload)
    except Exception as e:
        logger.error(f"Explain error: {str(e)}", exc_info=True)
        raise

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    
    logger.info(f"Starting FactoryGuard API | {HOST}:{PORT} | Debug: {DEBUG_MODE}")
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)