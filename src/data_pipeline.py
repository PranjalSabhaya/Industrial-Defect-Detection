import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR

AUTOTUNE = tf.data.AUTOTUNE

def load_dataset(directory,shuffle=True):
    raw_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )

    class_names = raw_ds.class_names

    ds = raw_ds.map(
        lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds, class_names