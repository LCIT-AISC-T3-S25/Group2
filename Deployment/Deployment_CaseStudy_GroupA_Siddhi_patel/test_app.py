from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "VGG Classifier with Metadata"}

def test_predict_valid():
    # Create a dummy image (small 224x224 RGB white)
    from PIL import Image
    import io
    image = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Sample valid metadata JSON string (example: array of numbers)
    metadata = json.dumps([25, 1, 22.5])  # adjust length & values to your real metadata shape

    response = client.post(
        "/predict/",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"metadata": metadata}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "predictions" in json_response
    assert isinstance(json_response["predictions"], list)

def test_predict_invalid_metadata():
    # Same dummy image
    from PIL import Image
    import io
    image = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Invalid JSON string for metadata
    metadata = "not a json"

    response = client.post(
        "/predict/",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"metadata": metadata}
    )
    # Your app might raise a 422 or 500 error because json.loads fails
    assert response.status_code in [400, 422, 500]
    assert "detail" in response.json() or "error" in response.json()

def test_predict_missing_file():
    metadata = json.dumps([25, 1, 22.5])
    response = client.post(
        "/predict/",
        data={"metadata": metadata}
    )
    assert response.status_code == 422  # Unprocessable Entity for missing file

def test_predict_missing_metadata():
    from PIL import Image
    import io
    image = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/predict/",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 422  # Missing required form field
