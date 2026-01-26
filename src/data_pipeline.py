import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR

AUTOTUNE = tf.data.AUTOTUNE

def load_dataset(directory, shuffle=True, cache=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        label_mode="int"   
    )

    ds = ds.map(
        lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
        num_parallel_calls=AUTOTUNE
    )

    if cache:
        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)

    return ds