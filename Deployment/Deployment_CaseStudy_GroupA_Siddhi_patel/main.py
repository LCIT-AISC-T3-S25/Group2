from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
from PIL import Image
import io
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()
model = load_model("model/vgg16_yelp_model.h5")

@app.get("/")
def read_root():
    return {"message": "VGG Classifier with Metadata"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...), metadata: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    metadata_array = np.array(json.loads(metadata)).reshape(1, -1)

    prediction = model.predict([image_array, metadata_array])
    return {"predictions": prediction.tolist()}

