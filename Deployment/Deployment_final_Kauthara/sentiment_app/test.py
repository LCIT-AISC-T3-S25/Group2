import os
import json
import pytest
from app import app  # Direct import since they're in same directory

# Initialize test client
client = app.test_client()

def test_valid_prediction():
    """Test happy path prediction"""
    payload = {"text": "I love sunny days!"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["confidence"], float)
    assert isinstance(data["top_contributing_words"], list)

def test_garbage_input():
    """Test garbage detection"""
    payload = {"text": "xkjb123!@#"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Garbage"

def test_short_text():
    """Test minimum length validation"""
    payload = {"text": "a"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400