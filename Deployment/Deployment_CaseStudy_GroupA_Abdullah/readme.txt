To run this app

step 1 - pip install -r requirements.txt
step 2 - python app.py
step 3 - run the curl
 request --  curl --location 'http://localhost:5000/predict' \
             --header 'Content-Type: application/json' \
             --data '{"text": "This movie was so so"}'
  Eg response -- 
      {
        "confidence_scores": {
          "negative": 0.017068728804588318,
          "neutral": 0.9746072292327881,
          "positive": 0.00832411739975214
        },
        "sentiment": "Neutral"
      }
