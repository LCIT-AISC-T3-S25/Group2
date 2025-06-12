import os
import sys
import pytest # type: ignore
from fastapi.testclient import TestClient

# Add root directory to Python path to avoid ModuleNotFoundError
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # This now works safely under any environment

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Image Classification API is running."}

def test_predict_endpoint_exists():
    response = client.post("/predict", files={"file": ("test.jpg", b"fake_image_data", "image/jpeg")})
    assert response.status_code == 500  # We expect error due to fake data
    assert "error" in response.json()

