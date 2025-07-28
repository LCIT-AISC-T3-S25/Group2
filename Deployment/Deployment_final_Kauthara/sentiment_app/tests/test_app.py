import requests

BASE_URL = "http://127.0.0.1:5000"
HEADERS = {"Content-Type": "application/json"}


def test_valid_prediction():
    payload = {"text": "I love sunny mornings in the park! 🌞🌳 #happy"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_contributing_words" in data
    print("test_valid_prediction passed")


def test_garbage_input():
    payload = {"text": "jfjfjf1234@#🤖"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Garbage"
    assert data["confidence"] == 0.0
    print("test_garbage_input passed")


def test_empty_text():
    payload = {"text": ""}
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    assert response.status_code == 400
    print("test_empty_text passed")


def test_missing_text_field():
    payload = {"message": "This will fail"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    assert response.status_code == 400
    print("test_missing_text_field passed")


def test_short_text():
    payload = {"text": "ok"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    assert response.status_code == 400
    print(" test_short_text passed")


if __name__ == "__main__":
    test_valid_prediction()
    test_garbage_input()
    test_empty_text()
    test_missing_text_field()
    test_short_text()
