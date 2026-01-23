import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models, layers
from src.config import IMG_SIZE, NUM_CLASSES

def build_model():
    base_model = EfficientNetB0(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.08, 0.08),
    ])

    inputs = layers.Input(shape=IMG_SIZE + (3,))

    x = data_augmentation(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model
