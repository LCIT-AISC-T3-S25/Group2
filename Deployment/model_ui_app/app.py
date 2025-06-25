import streamlit as st
import requests
import re
import string
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import words

st.set_page_config(page_title="Model Prediction UI", layout="wide")
st.title("🔍 Multi-Model Prediction Interface")

# Load English word list
try:
    english_vocab = set(words.words())
except:
    import nltk
    nltk.download('punkt')
    nltk.download('words')
    english_vocab = set(words.words())

# Smart input classifier
def classify_input(text, confidence_scores):
    tokens = word_tokenize(re.sub(f"[{string.punctuation}]", "", text.lower()))
    if len(tokens) > 0 and all(token not in english_vocab for token in tokens):
        return "Warning: Gibberish detected"
    if len(tokens) < 2:
        return "Warning: Input too short"
    sentiment = max(confidence_scores, key=confidence_scores.get)
    confidence = confidence_scores[sentiment]
    if sentiment.lower() == "neutral" and confidence > 0.95:
        return "Warning: High-confidence Neutral"
    elif confidence < 0.6:
        return "Warning: Emotionally unclear"
    else:
        return f"Sentiment: {sentiment}"

# Define tabs
tabs = st.tabs(["GRU Sentiment", "LSTM Sentiment", "CNN Image", "VGG Transfer Learning"])

# GRU Model - Port 5001
with tabs[0]:
    st.header("GRU Sentiment Analysis")
    gru_text = st.text_area("Enter text to analyze (GRU):", key="gru_text_area")
    if st.button("Predict using GRU", key="gru_button"):
        if gru_text:
            try:
                response = requests.post("http://localhost:5001/predict", json={"text": gru_text})
                if response.status_code == 200:
                    result = response.json()
                    message = classify_input(gru_text, result['confidence_scores'])
                    st.success(message)
                else:
                    st.error("Model error. Check if GRU container is running properly.")
            except Exception as e:
                st.error(f"Error connecting to GRU model: {e}")

# LSTM Model - Port 5002
with tabs[1]:
    st.header("LSTM Sentiment Analysis")
    lstm_text = st.text_area("Enter text to analyze (LSTM):", key="lstm_text_area")
    if st.button("Predict using LSTM", key="lstm_button"):
        if lstm_text:
            try:
                response = requests.post("http://localhost:5002/predict", json={"text": lstm_text})
                if response.status_code == 200:
                    result = response.json()
                    message = classify_input(lstm_text, result['confidence_scores'])
                    st.success(message)
                else:
                    st.error("Model error. Check if LSTM container is running properly.")
            except Exception as e:
                st.error(f"Error connecting to LSTM model: {e}")

# CNN Model - Port 5003
with tabs[2]:
    st.header("CNN Image Classification")
    cnn_image = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"], key="cnn_upload")
    if st.button("Classify with CNN", key="cnn_button"):
        if cnn_image:
            try:
                files = {"file": cnn_image.getvalue()}
                response = requests.post("http://localhost:5003/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result.get('prediction', 'No result returned')}")
                    st.info("🧠 Grad-CAM or saliency maps can be added here.")
                else:
                    st.error("CNN model error. Check if container is running.")
            except Exception as e:
                st.error(f"Error connecting to CNN model: {e}")

# VGG + Metadata Model - Port 5004 (metadata input removed)
with tabs[3]:
    st.header("VGG + Metadata Transfer Learning")
    vgg_image = st.file_uploader("Upload image for classification (VGG):", type=["jpg", "jpeg", "png"], key="vgg_upload")

    if st.button("Classify with VGG Transfer Learning", key="vgg_button"):
        if vgg_image:
            try:
                files = {
                    "file": ("image.png", vgg_image.getvalue(), vgg_image.type)
                }
                # Send empty list metadata as string
                data = {
                    "metadata": "[]"
                }

                response = requests.post("http://localhost:5004/predict", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Prediction successful!")
                    st.write("Prediction Output:", result)
                else:
                    st.error(f"VGG model error (status code {response.status_code}).")
                    try:
                        st.code(response.text)
                    except:
                        pass
            except Exception as e:
                st.error(f"❌ Error connecting to VGG model: {e}")