import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your FastAPI app (adjust the import based on your file name)
from app import app  # This should match your original app.py file

# Create the test client
client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data

def test_root_endpoint():
    """Test the root endpoint returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    # Check that it returns HTML content
    assert "text/html" in response.headers.get("content-type", "")

def test_generate_endpoint_basic():
    """Test the generate endpoint with default parameters"""
    # Use form data instead of JSON for your original app
    response = client.post("/generate", data={"num_images": 1})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "images" in data

def test_generate_endpoint_with_seed():
    """Test the generate endpoint with a specific seed"""
    response = client.post("/generate", data={"seed": 42, "num_images": 1})
    assert response.status_code == 200
    data = response.json()
    if data.get("success"):
        assert len(data["images"]) == 1
        assert data["images"][0]["seed"] == 42

def test_generate_endpoint_multiple_images():
    """Test generating multiple images"""
    response = client.post("/generate", data={"num_images": 3})
    assert response.status_code == 200
    data = response.json()
    if data.get("success"):
        assert len(data["images"]) == 3

def test_generate_endpoint_invalid_num_images():
    """Test that invalid number of images returns error"""
    response = client.post("/generate", data={"num_images": 15})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] == False
    assert "error" in data

def test_generate_endpoint_zero_images():
    """Test that zero images returns error"""
    response = client.post("/generate", data={"num_images": 0})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] == False
    assert "error" in data

def test_base64_image_format():
    """Test that generated images are in correct base64 format"""
    response = client.post("/generate", data={"seed": 123, "num_images": 1})
    assert response.status_code == 200
    data = response.json()
    
    if data.get("success"):
        # Check that image is in base64 format
        image_data = data["images"][0]["image"]
        assert image_data.startswith("data:image/png;base64,")
        
        # Verify base64 content exists
        base64_content = image_data.split(",")[1]
        assert len(base64_content) > 0