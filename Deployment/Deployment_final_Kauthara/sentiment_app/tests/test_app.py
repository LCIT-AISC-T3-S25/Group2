import os
import sys
import json
import pytest

# Ensure root directory is in sys.path so app.py can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

client = app.test_client()

def test_valid_prediction():
    payload = {"text": "I love sunny mornings in the park! 🌞🌳 #happy"}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_contributing_words" in data

def test_garbage_input():
    payload = {"text": "jfjfjf1234@#🤖"}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] == "Garbage"
    assert data["confidence"] == 0.0

def test_empty_text():
    payload = {"text": ""}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 400

def test_missing_text_field():
    payload = {"message": "This will fail"}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 400

def test_short_text():
    payload = {"text": "ok"}
    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 400