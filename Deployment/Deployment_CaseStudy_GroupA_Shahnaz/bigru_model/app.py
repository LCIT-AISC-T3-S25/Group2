from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import yaml
import os

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
word_lenth=10

model = load_model("BiGRU_model.h5")
vocab_size = config["model"]["vocab_size"]
maxlen = config["model"]["maxlen"]
port = config["server"]["port"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sequences = data.get("sequences")  # list of tokenized sequences
    sequences = pad_sequences(sequences, maxlen=maxlen)
    preds = model.predict(sequences)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
