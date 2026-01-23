import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from src.data_pipeline import load_dataset
from src.config import VAL_DIR,MODEL_PATH

model = tf.keras.models.load_model(MODEL_PATH)
val_ds,class_names = load_dataset(VAL_DIR, shuffle=False)

y_true, y_pred = [], []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred,target_names=class_names))
