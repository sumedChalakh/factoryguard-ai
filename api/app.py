import joblib
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path
import sys
import os
import json
from werkzeug.exceptions import HTTPException
from jsonschema import Draft202012Validator

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import create_features
from src.utils import explain_with_shap

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
app = Flask(__name__)
DEBUG_MODE = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
app.config["PROPAGATE_EXCEPTIONS"] = DEBUG_MODE

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
SCHEMA_PATH = BASE_DIR / "api" / "schema.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(PREPROCESSOR_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
    API_SCHEMAS = json.load(schema_file)


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

# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    return jsonify({
        "error": error.name,
        "message": error.description
    }), error.code


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
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
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Empty JSON payload"}), 400

    validation_error = validate_request_payload(data, "predict_request")
    if validation_error:
        return validation_error

    X = preprocess_input(data)
    X_scaled = scaler.transform(X)

    failure_prob = model.predict_proba(X_scaled)[0][1]

    response_payload = {
        "failure_probability": round(float(failure_prob), 6)
    }

    validate_response_payload(response_payload, "predict_response")
    return jsonify(response_payload)

# ------------------------------------------------------------------
# PREDICT + SHAP EXPLANATION
# ------------------------------------------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Empty JSON payload"}), 400

    validation_error = validate_request_payload(data, "explain_request")
    if validation_error:
        return validation_error

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
    return jsonify(response_payload)

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=DEBUG_MODE)