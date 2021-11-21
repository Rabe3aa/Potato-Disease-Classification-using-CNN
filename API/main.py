from fastapi import FastAPI, UploadFile, File
from starlette.routing import Host
import uvicorn
import numpy as np
from PIL  import Image
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("../Models/1.0")
CLASSES = ['Early Blight', 'Late Blight', 'Healthy']

def bytes_to_img(data):
    img_array = np.array(Image.open(BytesIO(data)))
    return img_array

@app.get("/starter")
async def starter():
    return "I'm alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = bytes_to_img(await file.read())
    img = np.expand_dims(img,0)
    prediction = model.predict(img)
    label = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return label, confidence

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=9999)