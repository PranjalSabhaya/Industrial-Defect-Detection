import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

from src.utils import load_config
from src.data_pipeline import load_dataset


def run_evaluation(config_path: str):
    # ---- load config ----
    config = load_config(config_path)

    val_dir = config["data"]["val_dir"]
    img_size = tuple(config["model"]["img_size"])
    batch_size = config["training"]["batch_size"]
    model_path = config["model"]["model_path"]

    # ---- load model ----
    model = tf.keras.models.load_model(model_path)

    # ---- load validation dataset ----
    val_ds = load_dataset(
        directory=val_dir,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # ---- load class names ----
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    # ---- evaluation ----
    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print(report)
    return report


if __name__ == "__main__":
    run_evaluation("config/local.yaml")
