# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import yaml
import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load tokenizer
with open("tokenizer_biGRU.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("BiGRU_model.h5", compile=False)

# Read values from config
max_len = config["model"].get("maxlen", 50)
trunc_type = config["model"].get("trunc_type", "post")
port = config["server"].get("port", 5000)

# Hardcoded class labels
label_classes = ['Negative', 'Neutral', 'Positive']

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "BiGRU Prediction API is running. Use POST /predict with JSON {'text': 'your input'}"

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_len, truncating=trunc_type)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.get_json(force=True).get("text")
        if not text:
            return jsonify({"error": "Missing 'text' key in request JSON"}), 400

        padded_input = preprocess_text(text)
        y_prob = model.predict(padded_input)[0]
        y_index = int(np.argmax(y_prob))
        predicted_label = label_classes[y_index]

        return jsonify({
            "sentiment": predicted_label,
            "confidence_scores": {
                label: float(score)
                for label, score in zip(label_classes, y_prob)
            }
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting BiGRU prediction service...")
    app.run(host="0.0.0.0", port=port)
