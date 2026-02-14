import cv2
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


def set_seed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_image(img, target_size):
    if img is None:
        raise ValueError("Invalid image: None")

    # Grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # RGBA → RGB
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    if img.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB)")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    img = cv2.resize(img, target_size)
    return img


def predict(model, config: dict, image_array):
    """
    Predict using preloaded model.
    """

    if image_array is None:
        raise ValueError("No image provided")

    # ----- config -----
    img_size = tuple(config["model"]["img_size"])
    confidence_threshold = config["inference"]["confidence_threshold"]

    # ----- validate + resize -----
    img = validate_image(image_array, img_size)

    # ----- preprocess -----
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    # ----- prediction -----
    preds = model.predict(img, verbose=0)

    label = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if confidence < confidence_threshold:
        return {
            "status": "uncertain",
            "confidence": confidence
        }

    return {
        "status": "success",
        "label": label,
        "confidence": confidence
    }
