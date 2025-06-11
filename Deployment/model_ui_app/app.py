import streamlit as st
import requests

# --- Updated API URLs matching your docker port mappings ---
GRU_API_URL = "http://localhost:5004/predict"   # GRU container on port 5004
LSTM_API_URL = "http://localhost:5002/predict"  # LSTM container on port 5002
CNN_API_URL = "http://localhost:5003/predict"   # CNN container on port 5003

st.title("Multi-Model Prediction UI")

tabs = st.tabs(["GRU Model", "LSTM Model", "CNN Model"])

# -------- GRU Model Tab --------
with tabs[0]:
    st.header("GRU Model Demo")
    gru_input = st.text_area("Enter text for GRU model prediction")
    if st.button("Run GRU Model", key="gru_btn"):
        if gru_input.strip():
            try:
                response = requests.post(GRU_API_URL, json={"text": gru_input})
                if response.ok:
                    result = response.json()
                    st.success(f"Prediction: {result.get('prediction', 'No prediction in response')}")
                    st.json(result)
                else:
                    st.error(f"GRU Model API error: {response.status_code}")
            except Exception as e:
                st.error(f"Error calling GRU model API: {e}")
                st.info("Make sure the GRU container is running and mapped to port 5004.")
        else:
            st.warning("Please enter some text")

# -------- LSTM Model Tab --------
with tabs[1]:
    st.header("LSTM Model Demo")
    lstm_input = st.text_area("Enter text for LSTM model prediction")
    if st.button("Run LSTM Model", key="lstm_btn"):
        if lstm_input.strip():
            try:
                response = requests.post(LSTM_API_URL, json={"text": lstm_input})
                if response.ok:
                    result = response.json()
                    st.success(f"Prediction: {result.get('prediction', 'No prediction in response')}")
                    st.json(result)
                else:
                    st.error(f"LSTM Model API error: {response.status_code}")
            except Exception as e:
                st.error(f"Error calling LSTM model API: {e}")
                st.info("Make sure the LSTM container is running and mapped to port 5002.")
        else:
            st.warning("Please enter some text")

# -------- CNN Model Tab --------
with tabs[2]:
    st.header("CNN Model Demo")
    uploaded_file = st.file_uploader("Upload an image for CNN model prediction", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Run CNN Model", key="cnn_btn"):
        if uploaded_file is not None:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(CNN_API_URL, files=files)
                if response.ok:
                    result = response.json()
                    st.success(f"Prediction: {result.get('prediction', 'No prediction in response')}")
                    st.json(result)
                else:
                    st.error(f"CNN Model API error: {response.status_code}")
            except Exception as e:
                st.error(f"Error calling CNN model API: {e}")
                st.info("Make sure the CNN container is running and mapped to port 5003.")
        else:
            st.warning("Please upload an image file")
