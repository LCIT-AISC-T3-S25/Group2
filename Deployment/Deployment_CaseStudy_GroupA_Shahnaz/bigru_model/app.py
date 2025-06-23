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
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Configuration file 'config.yaml' not found.")

# Load tokenizer
try:
    with open("tokenizer_biGRU.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    raise RuntimeError("Failed to load tokenizer: " + str(e))

# Load model
try:
    model = load_model("BiGRU_model.h5", compile=False)
except Exception as e:
    raise RuntimeError("Failed to load model: " + str(e))

# Read values from config
max_len = config["model"].get("maxlen", 50)
trunc_type = config["model"].get("trunc_type", "post")
port = config["server"].get("port", 5000)

# Define hardcoded label classes (in the same order the model was trained on)
label_classes = ['Negative', 'Neutral', 'Positive']

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "BiGRU Prediction API is running. Use POST /predict with JSON {'text': 'your input'}"

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, truncating=trunc_type)
    return padded

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' key in JSON body"}), 400

        text = data["text"]
        padded_input = preprocess_text(text)
        y_prob = model.predict(padded_input)[0]
        y_index = int(np.argmax(y_prob))
        predicted_label = label_classes[y_index]

        confidence_scores = {
            label: float(score)
            for label, score in zip(label_classes, y_prob)
        }

        return jsonify({
            "sentiment": predicted_label,
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting BiGRU prediction service...")
    app.run(host="0.0.0.0", port=port)
