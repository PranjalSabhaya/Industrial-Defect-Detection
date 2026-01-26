import json
import cv2
import random
import tensorflow as tf
import numpy as np
import glob
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config import MODEL_PATH, IMG_SIZE

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

model = tf.keras.models.load_model(MODEL_PATH)

def validate_image(img, target_size=IMG_SIZE):
    if img is None:
        raise ValueError("Invalid image: None")

    # If grayscale → convert to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # If RGBA → drop alpha channel
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # If not RGB even now → reject
    if img.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB)")

    # Ensure correct dtype
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Resize to model input size
    img = cv2.resize(img, target_size)

    return img


def predict(image_path):
    img = cv2.imread(image_path)
    img = validate_image(img)
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img,verbose=0)

    label = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return label, confidence
