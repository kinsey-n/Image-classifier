from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
from backend.model import ImageClassifier
import os
import numpy as np
import tensorflow as tf
import pandas
from keras import layers
from keras.utils import image_dataset_from_directory 


app = FastAPI(title="Image Classifier API")
classifier = ImageClassifier("models/image_classifier/weights.weights.h5")



class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images supported")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return classifier.predict(image)
