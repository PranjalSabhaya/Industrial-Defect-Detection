import json
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config import MODEL_PATH, IMG_SIZE

model = tf.keras.models.load_model(MODEL_PATH)

def predict(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img,verbose=0)

    label = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return label, confidence

sample_path = glob.glob("../data/raw/NEU-DET/validation/*/*.jpg")[0]

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

if __name__ == "__main__":
    label, confidence = predict(sample_path)
    print(f"Predicted class: {CLASS_NAMES[label]}, Confidence: {confidence:.4f}")
