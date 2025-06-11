import pytest
from unittest.mock import patch, MagicMock
import json

# Patch model and config before importing the app
@pytest.fixture
def client():
    with patch("builtins.open"), \
         patch("yaml.safe_load") as mock_yaml_load, \
         patch("tensorflow.keras.models.load_model") as mock_load_model:

        # Mock YAML config
        mock_yaml_load.return_value = {
            "model": {"vocab_size": 10000, "maxlen": 10},
            "server": {"port": 5000}
        }

        # Mock model prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.1, 0.9]]
        mock_load_model.return_value = mock_model

        # Import the app after mocking
        from app import app  # assuming your original code is in app.py

        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

def test_predict_endpoint(client):
    # Sample input
    test_data = {
        "sequences": [[1, 2, 3, 4]]
    }

    response = client.post(
        "/predict",
        data=json.dumps(test_data),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
