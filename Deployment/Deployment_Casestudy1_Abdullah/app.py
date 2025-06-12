# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

max_len = 250                 # Maximum length of sequences after padding
trunc_type = "post"           # Truncate sequences at the end

# Load model
model = load_model("BiLSTM_model.h5", compile=False)

# Load tokenizer
with open("tokenizer_lstm.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    labelencoder = pickle.load(f)

app = Flask(__name__)

def preprocess_text(text):
    train_word_sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(train_word_sequence, maxlen=max_len, truncating=trunc_type)
    return padded

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    x_input = preprocess_text(text)
    
    # Model predicts probabilities for 3 classes
    y_prob = model.predict(x_input)[0]  # shape: (3,)
    y_class_index = np.argmax(y_prob)
    print(y_class_index, y_prob)
    # Decode predicted class label
    predicted_label = labelencoder.inverse_transform([y_class_index])[0]
    print(predicted_label)
    return jsonify({
        "sentiment": predicted_label,
        "confidence_scores": {
            "positive": float(y_prob[labelencoder.transform(['Positive'])[0]]),
            "neutral": float(y_prob[labelencoder.transform(['Neutral'])[0]]),
            "negative": float(y_prob[labelencoder.transform(['Negative'])[0]])
        }
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)