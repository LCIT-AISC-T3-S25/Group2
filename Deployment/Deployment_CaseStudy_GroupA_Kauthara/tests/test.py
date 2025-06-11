from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Image Classification API is running."}

def test_predict_endpoint_exists():
    response = client.post("/predict", files={"file": ("test.jpg", b"fake_image_data", "image/jpeg")})
    # Because we're passing fake image bytes, we expect a 500 error
    assert response.status_code == 500
    assert "error" in response.json()
