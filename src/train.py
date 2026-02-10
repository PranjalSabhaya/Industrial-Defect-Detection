import json
import tensorflow as tf
from tensorflow.keras import callbacks

from src.data_pipeline import load_dataset
from src.model import build_model


def save_class_names(train_dir, img_size, batch_size):
    class_names = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size
    ).class_names

    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    return class_names


def get_callbacks(model_path):
    return [
        callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6
        )
    ]


def train_base_model(config, train_ds, val_ds):
    print("\n===== PHASE 1: BASE MODEL TRAINING =====")

    model = build_model(
        img_size=tuple(config["model"]["img_size"]),
        num_classes=config["model"]["num_classes"]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config["training"]["learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=get_callbacks(config["model"]["model_path"])
    )

    return model, history


def should_fine_tune(history, threshold=0.90):
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return best_val_acc >= threshold


def fine_tune_model(config, model, train_ds, val_ds):
    print("\n===== PHASE 2: FINE-TUNING =====")

    base_model = model.layers[1]  
    base_model.trainable = True

    fine_tune_at = int(len(base_model.layers) * 0.80)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config["training"]["fine_tune_learning_rate"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=get_callbacks(config["model"]["model_path"])
    )

    return model


def train_model(config: dict):
    # Extract config
    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]

    img_size = tuple(config["model"]["img_size"])
    batch_size = config["training"]["batch_size"]
    model_path = config["model"]["model_path"]

    # Save class names once
    save_class_names(train_dir, img_size, batch_size)

    # Load datasets
    train_ds = load_dataset(
        train_dir,
        img_size=img_size,
        batch_size=batch_size
    )

    val_ds = load_dataset(
        val_dir,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Phase 1
    model, history = train_base_model(config, train_ds, val_ds)

    # Phase 2 (conditional)
    if should_fine_tune(history):
        model = fine_tune_model(config, model, train_ds, val_ds)
    else:
        print("Skipping fine-tuning — base model not stable enough")

    # Save final model
    model.save(model_path, include_optimizer=False)
    print(f"\nModel saved to {model_path}")
