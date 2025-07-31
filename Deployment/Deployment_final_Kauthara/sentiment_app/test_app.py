import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()

def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sentiment Analysis API is running" in response.data
    print("test_home_route passed!")

def test_test_ui_route(client):
    response = client.get("/test-ui")
    assert response.status_code == 200
    assert b"Test Sentiment Analysis" in response.data
    print("test_test_ui_route passed!")

def test_missing_text_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()
    print("test_missing_text_field passed!")

def test_valid_prediction(client):
    payload = {"text": "I love sunny days!"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data.get("confidence", None), float)
    assert isinstance(data.get("top_contributing_words", None), list)
    print("test_valid_prediction passed!")
