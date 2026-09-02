import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
import numpy as np 
from PIL import Image

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

class ImageClassifier:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image: Image.Image) -> dict:
        image = image.resize((150, 150)).convert("RGB")
        arr = np.expand_dims(np.array(image) / 255.0, axis=0)
        preds = self.model.predict(arr)[0]
        idx = int(np.argmax(preds))
        return {"label": CLASS_NAMES[idx], "confidence": float(preds[idx])}

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

print("Step 1: loading datasets...", flush=True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train_ = image_dataset_from_directory(
    'C:/Users/kinse/Desktop/Image classifier/seg_train',
    labels='inferred',
    label_mode='int',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    'C:/Users/kinse/Desktop/Image classifier/seg_test',
    labels='inferred',
    label_mode='int',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

print("Step 2: datasets loaded, building pipeline...", flush=True)

ds_train = ds_train_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid_.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)

print("Step 3: building model...", flush=True)
classifier = ImageClassifier(r"C:\Users\kinse\Desktop\Image classifier\models\image_classifier")
print("Step 4: starting training (this will take a while)...", flush=True)
classifier.train(ds_train, ds_valid, epochs=50, save_path="models/image_classifier/weights.weights.h5")

print("Step 5: training finished, saving...", flush=True)

save_dir = "models/image_classifier"
os.makedirs(save_dir, exist_ok=True)   # <-- creates the folder if it doesn't exist
print(f"Step 6: '{save_dir}' now exists: {os.path.exists(save_dir)}", flush=True)