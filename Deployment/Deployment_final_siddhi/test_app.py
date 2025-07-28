import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app, model, DEVICE

client = TestClient(app)

def test_root_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "healthy"
    assert isinstance(json_resp["model_loaded"], bool)
    assert json_resp["device"] == str(DEVICE)

def test_generate_default():
    response = client.post("/generate")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["images"], list)
    assert len(data["images"]) == 1
    assert data["images"][0]["image"].startswith("data:image/png;base64,")
    assert isinstance(data["images"][0]["seed"], int)

def test_generate_with_seed_and_multiple_images():
    response = client.post("/generate", data={"seed": "42", "num_images": "3"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["images"]) == 3
    seeds = [img["seed"] for img in data["images"]]
    assert seeds == [42, 43, 44]

@pytest.mark.parametrize("num_images", ["0", "11", "-5", "20"])
def test_generate_invalid_num_images(num_images):
    response = client.post("/generate", data={"num_images": num_images})
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Number of images" in data["error"]

def test_generate_model_not_loaded():
    with patch("main.model", None):
        response = client.post("/generate")
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "Model not loaded" in data["error"]

@pytest.mark.parametrize("payload", [
    {"seed": "not_an_int"},
    {"num_images": "not_an_int"},
    {}
])
def test_generate_invalid_form_data(payload):
    response = client.post("/generate", data=payload)
    assert response.status_code == 422
