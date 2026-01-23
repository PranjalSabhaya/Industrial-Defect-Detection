import json
import tensorflow as tf
from tensorflow.keras import callbacks
from src.config import EPOCHS, LEARNING_RATE, MODEL_PATH,TRAIN_DIR,VAL_DIR
from src.data_pipeline import load_dataset
from src.model import build_model

def get_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6
        )
    ]

def train_base_model(train_ds, val_ds):
    print("\n===== PHASE 1: BASE MODEL TRAINING =====")

    model = build_model()  # backbone frozen inside model.py

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks()
    )

    return model, history

def should_fine_tune(history, threshold=0.90):
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    return best_val_acc >= threshold

def fine_tune_model(model, train_ds, val_ds):
    print("\n===== PHASE 2: FINE-TUNING =====")

    base_model = model.layers[1]  # EfficientNet

    base_model.trainable = True

    FINE_TUNE_AT = int(len(base_model.layers) * 0.80)

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks()
    )

    return model

def main():
    train_ds,class_names = load_dataset(TRAIN_DIR)
    val_ds,_   = load_dataset(VAL_DIR, shuffle=False)

    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    model, history = train_base_model(train_ds, val_ds)

    if should_fine_tune(history):
        model = fine_tune_model(model, train_ds, val_ds)
    else:
        print("Skipping fine-tuning — base model not stable enough")

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
