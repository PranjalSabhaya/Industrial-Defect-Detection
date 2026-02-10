import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(
    directory: str,
    img_size: tuple,
    batch_size: int,
    shuffle: bool = True,
    cache: bool = True):
    

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
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
