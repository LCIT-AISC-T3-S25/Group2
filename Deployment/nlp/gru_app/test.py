import unittest
from unittest.mock import patch, MagicMock
import json
from flask import Flask

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        # Patch open, yaml.safe_load, and model loader before importing the app
        self.patcher_open = patch("builtins.open", create=True)
        self.patcher_yaml = patch("yaml.safe_load")
        self.patcher_model = patch("tensorflow.keras.models.load_model")

        self.mock_open = self.patcher_open.start()
        self.mock_yaml = self.patcher_yaml.start()
        self.mock_load_model = self.patcher_model.start()

        # Mock config
        self.mock_yaml.return_value = {
            "model": {"vocab_size": 10000, "maxlen": 10},
            "server": {"port": 5000}
        }

        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [[0.3, 0.7]]
        self.mock_load_model.return_value = self.mock_model

        # Import the app AFTER mocks
        from app import app
        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        self.patcher_open.stop()
        self.patcher_yaml.stop()
        self.patcher_model.stop()

    def test_predict_valid_input(self):
        response = self.app.post(
            "/predict",
            data=json.dumps({"sequences": [[1, 2, 3, 4]]}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predictions", data)
        self.assertEqual(data["predictions"], [[0.3, 0.7]])

    def test_predict_missing_sequences(self):
        response = self.app.post(
            "/predict",
            data=json.dumps({"wrong_key": [[1, 2]]}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # Still 200
        data = response.get_json()
        self.assertIn("predictions", data)
        # The model is still called with None, but pad_sequences will pad it
        self.assertIsInstance(data["predictions"], list)

    def test_predict_invalid_json(self):
        response = self.app.post(
            "/predict",
            data="not-a-json",
            content_type="application/json"
        )
        # Should raise 400 from Flask
        self.assertEqual(response.status_code, 400)

    def test_model_prediction_exception(self):
        # Force model.predict to raise exception
        self.mock_model.predict.side_effect = Exception("Prediction error")
        response = self.app.post(
            "/predict",
            data=json.dumps({"sequences": [[1, 2, 3]]}),
            content_type="application/json"
        )
        # Flask will raise 500 by default on unhandled exceptions
        self.assertEqual(response.status_code, 500)

if __name__ == "__main__":
    unittest.main()
