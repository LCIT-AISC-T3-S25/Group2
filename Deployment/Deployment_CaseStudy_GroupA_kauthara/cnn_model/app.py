from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np
import joblib
import io

app = FastAPI()

# Load model and label encoder
model = load_model("cnn_best_model1.h5")
label_encoder = joblib.load("label_encoder1.pkl")

# Preprocessing function — resize to 224x224, normalize, expand dims
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  #  Match model training size
    image_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)  # shape: (1, 224, 224, 3)

@app.get("/")
def read_root():
    return {"message": "Image Classification API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        prediction = model.predict(image)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        return JSONResponse(content={"prediction": predicted_class})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
