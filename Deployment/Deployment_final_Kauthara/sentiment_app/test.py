import os
import sys
import json
import pytest
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent if "tests" in str(Path(__file__)) else Path(__file__).parent

# Add root directory to Python path
sys.path.insert(0, str(ROOT_DIR))

# Set the working directory to the root directory
os.chdir(ROOT_DIR)

# Now import the app after setting paths
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