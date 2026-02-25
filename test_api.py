from api.app import app


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