import os
import pathlib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("C:\\Users\\pc\\Desktop\\disease_potato\\potato-disease-classification-main\\saved_models\\1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

uploads='uploads'
uploads_dir=pathlib.Path(os.getcwd(),uploads)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    
    return image

@app.post("/predict")
async def predict(
    file: UploadFile =File()
):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png","JPG","JPEG","PNG")
    if not extension:
        return "Image must be jpg , jpeg or png format!"
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) 
    for x in predictions:
        print(x)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
# @app.post("/predict")
# async def predict():
#     return {
#         'class': "ABC",
#         'confidence': 99
#     }
    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5050, log_level='info')

