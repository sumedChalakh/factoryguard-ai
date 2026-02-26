import pandas as pd
import pytest
import api.app as app_module


app = app_module.app


class DummyScaler:
    feature_names_in_ = ["temperature", "vibration", "pressure"]

    def transform(self, X):
        return X[self.feature_names_in_].to_numpy(dtype=float)


class DummyModel:
    def predict_proba(self, X):
        result = []
        for row in X:
            mean_value = float(sum(row) / len(row)) if len(row) else 0.0
            score = max(0.0, min(1.0, mean_value / 200.0))
            result.append([1.0 - score, score])
        return result


@pytest.fixture(autouse=True)
def patch_model_and_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "model", DummyModel(), raising=False)
    monkeypatch.setattr(app_module, "scaler", DummyScaler(), raising=False)
    monkeypatch.setattr(
        app_module,
        "preprocess_input",
        lambda data: pd.DataFrame([
            {
                "temperature": float(data["temperature"]),
                "vibration": float(data["vibration"]),
                "pressure": float(data["pressure"]),
            }
        ]),
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "explain_with_shap",
        lambda model, X_row: {
            "base_value": 0.5,
            "shap_values": {col: 0.0 for col in X_row.columns},
        },
        raising=False,
    )


def _valid_payload():
    return {
        "temperature": 60,
        "vibration": 29,
        "pressure": 102,
    }


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_json()
    assert body == {"status": "ok"}


def test_predict_valid_payload_returns_probability():
    client = app.test_client()
    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert "failure_probability" in body
    assert isinstance(body["failure_probability"], float)
    assert 0.0 <= body["failure_probability"] <= 1.0


def test_explain_valid_payload_returns_expected_shape():
    client = app.test_client()
    response = client.post("/explain", json=_valid_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert "failure_probability" in body
    assert "base_value" in body
    assert "shap_values" in body
    assert isinstance(body["shap_values"], dict)
    assert len(body["shap_values"]) > 0


def test_predict_empty_payload_returns_400():
    client = app.test_client()
    response = client.post("/predict", json={})
    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body


def test_predict_missing_required_field_returns_400():
    client = app.test_client()
    response = client.post("/predict", json={"temperature": 60, "vibration": 29})
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Invalid request payload"


def test_predict_wrong_type_returns_400():
    client = app.test_client()
    response = client.post("/predict", json={"temperature": "hot", "vibration": 29, "pressure": 102})
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Invalid request payload"


def test_liveness_and_readiness_endpoints():
    client = app.test_client()

    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.get_json()["status"] == "alive"

    ready = client.get("/health/ready")
    assert ready.status_code in (200, 503)


def test_docs_and_metrics_endpoints():
    client = app.test_client()

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "text/html" in docs.content_type

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "openapi" in openapi.get_json()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "factoryguard_requests_total" in metrics.get_data(as_text=True)


def test_predict_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "topsecret")
    client = app.test_client()

    unauthorized = client.post("/predict", json=_valid_payload())
    assert unauthorized.status_code == 401

    authorized = client.post("/predict", json=_valid_payload(), headers={"X-API-Key": "topsecret"})
    assert authorized.status_code == 200