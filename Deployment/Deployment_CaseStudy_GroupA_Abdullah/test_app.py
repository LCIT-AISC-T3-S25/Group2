import json
from app import app

def test_sentiment_prediction():
    tester = app.test_client()
    
    # Example test input
    response = tester.post(
        "/predict",
        data=json.dumps({"text": "I love this product!"}),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()
    print(data)
    assert "sentiment" in data
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
