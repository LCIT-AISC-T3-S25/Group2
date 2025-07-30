import os
import sys
import json
import pytest
from pathlib import Path

# --- Critical Path Fixes ---
# 1. Get the absolute path to the sentiment_app directory
APP_DIR = Path(__file__).parent

# 2. Set the working directory to ensure app.py finds config.yaml
os.chdir(APP_DIR)

# 3. Verify critical files exist (debugging)
print(f"\n[DEBUG] Current directory: {os.getcwd()}")
print(f"[DEBUG] Files here: {os.listdir(APP_DIR)}\n")

# 4. Import app AFTER setting paths
from app import app

# --- Test Setup ---
client = app.test_client()

# --- Test Cases ---
def test_valid_prediction():
    """Test valid sentiment prediction"""
    payload = {"text": "I love sunny mornings in the park! 🌞🌳 #happy"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_contributing_words" in data

def test_garbage_input():
    """Test garbage input detection"""
    payload = {"text": "jfjfjf1234@#🤖"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] == "Garbage"
    assert data["confidence"] == 0.0

def test_empty_text():
    """Test empty input handling"""
    payload = {"text": ""}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400

def test_missing_text_field():
    """Test missing required field"""
    payload = {"message": "This will fail"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400

def test_short_text():
    """Test short text rejection"""
    payload = {"text": "ok"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400