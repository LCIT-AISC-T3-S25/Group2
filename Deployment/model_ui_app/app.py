import streamlit as st
import requests

st.set_page_config(page_title="Model Prediction UI", layout="wide")

st.title("🔍 Multi-Model Prediction Interface")

# Define tabs for 4 models
tabs = st.tabs(["GRU Sentiment", "LSTM Sentiment", "CNN Image", "VGG Transfer Learning"])

# GRU Model - Port 5001
with tabs[0]:
    st.header("GRU Sentiment Analysis")
    gru_text = st.text_area("Enter text to analyze (GRU):", key="gru_text_area")
    if st.button("Predict using GRU", key="gru_button"):
        if gru_text:
            try:
                response = requests.post("http://localhost:8501/predict", json={"text": gru_text})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['prediction']}")
                    st.info(" Interpretability feature can be added here (e.g., attention weights)")
                else:
                    st.error("Model error. Check if GRU container is running properly.")
            except Exception as e:
                st.error(f"Error connecting to GRU model: {e}")

# LSTM Model - Port 5002
with tabs[1]:
    st.header(" LSTM Sentiment Analysis")
    lstm_text = st.text_area("Enter text to analyze (LSTM):", key="lstm_text_area")
    if st.button("Predict using LSTM", key="lstm_button"):
        if lstm_text:
            try:
                response = requests.post("http://localhost:5002/predict", json={"text": lstm_text})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['prediction']}")
                    st.info("Interpretability feature can be added here (e.g., LIME)")
                else:
                    st.error("Model error. Check if LSTM container is running properly.")
            except Exception as e:
                st.error(f"Error connecting to LSTM model: {e}")

# CNN Model - Port 8000
with tabs[2]:
    st.header("CNN Image Classification")
    cnn_image = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"], key="cnn_upload")
    if st.button("Classify with CNN", key="cnn_button"):
        if cnn_image:
            try:
                files = {"file": cnn_image.getvalue()}
                response = requests.post("http://localhost:8000/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['class']}")
                    st.info("Grad-CAM or saliency maps can go here.")
                else:
                    st.error("CNN model error. Check if container is running.")
            except Exception as e:
                st.error(f"Error connecting to CNN model: {e}")

# VGG + Metadata Model - Port 8001
with tabs[3]:
    st.header("VGG + Metadata Transfer Learning")
    vgg_image = st.file_uploader("Upload image for classification (VGG):", type=["jpg", "jpeg", "png"], key="vgg_upload")
    if st.button("Classify with VGG Transfer Learning", key="vgg_button"):
        if vgg_image:
            try:
                files = {"file": vgg_image.getvalue()}
                response = requests.post("http://localhost:8001/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['class']}")
                    st.info("Explanation of VGG + Metadata combination can go here.")
                else:
                    st.error("VGG model error. Check if container is running.")
            except Exception as e:
                st.error(f"Error connecting to VGG model: {e}")
