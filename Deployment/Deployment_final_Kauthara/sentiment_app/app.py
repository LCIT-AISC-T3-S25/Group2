import os
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import yaml
from lime.lime_text import LimeTextExplainer
from keras.utils import register_keras_serializable
import tensorflow as tf

# ========== Load config ==========
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

host = config["app"]["host"]
port = config["app"]["internal_port"]
max_len = config["app"]["max_len"]

# ========== Load tokenizer ==========
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

# ========== Load label classes ==========
label_classes = np.load("label_classes.npy", allow_pickle=True)

# ========== Register Custom Layer ==========
@register_keras_serializable()
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = self.pos_emb(positions)
        return x + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim
        })
        return config

# ========== Load model ==========
model = load_model("sentiment_transformer_model.keras", compile=False)

# ========== Initialize LIME ==========
explainer = LimeTextExplainer(class_names=label_classes.tolist())

def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return model.predict(padded)

# ========== App ==========
app = Flask(__name__)

# ========== Combined Endpoint ==========
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No input text provided."}), 400

        text = data["text"].strip()
        if not text or len(text) < 3:
            return jsonify({"error": "Input text is too short."}), 400

        # Garbage Detection
        sequence = tokenizer.texts_to_sequences([text])
        tokens = sequence[0]
        unique_tokens = set(tokens)

        if len(tokens) < 3 or (len(unique_tokens) <= 1 and tokens.count(tokens[0]) > 1):
            return jsonify({
                "prediction": "Garbage",
                "confidence": 0.0,
                "top_contributing_words": [],
                "note": "Input flagged as garbage or meaningless."
            }), 200

        # Prediction
        padded = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
        predictions = model.predict(padded)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = label_classes[predicted_index]
        confidence = float(predictions[predicted_index])

        # LIME Explanation
        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=10,
            top_labels=1
        )
        label_index = explanation.top_labels[0]
        explanation_list = explanation.as_list(label=label_index)

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "top_contributing_words": explanation_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== Root Browser Route ========== 
@app.route("/", methods=["GET"])
def home():
    return "<h2>Sentiment Analysis API is running. Use POST /predict with JSON like {'text': 'your message'}.</h2>"

# ========== Simple UI for Manual Testing ==========
@app.route("/test-ui", methods=["GET"])
def test_ui():
    return '''
    <h2>Test Sentiment Analysis</h2>
    <form method="post" action="/predict" enctype="application/json" onsubmit="submitForm(event)">
        <label for="text">Enter text:</label><br>
        <input type="text" id="text" name="text" size="60"><br><br>
        <input type="submit" value="Analyze">
    </form>
    <pre id="result"></pre>
    <script>
    async function submitForm(event) {
        event.preventDefault();
        const text = document.getElementById("text").value;
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text})
        });
        const result = await response.json();
        document.getElementById("result").textContent = JSON.stringify(result, null, 2);
    }
    </script>
    '''

# ========== Run ========== 
if __name__ == "__main__":
    app.run(host=host, port=port)