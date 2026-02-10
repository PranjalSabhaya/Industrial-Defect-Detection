import json
import tensorflow as tf
import numpy as np

from src.utils import load_config
from src.data_pipeline import load_dataset
from src.model import build_model


def validate_data_pipeline(config, class_names):
    print("🔹 Checking data pipeline...")

    train_ds = load_dataset(
        directory=config["data"]["train_dir"],
        img_size=tuple(config["model"]["img_size"]),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_ds = load_dataset(
        directory=config["data"]["val_dir"],
        img_size=tuple(config["model"]["img_size"]),
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    print("✔ Class names:", class_names)

    for images, labels in train_ds.take(1):
        assert images.shape[1:] == tuple(config["model"]["img_size"]) + (3,), \
            "❌ Image shape mismatch"
        assert len(labels.shape) == 1, "❌ Labels shape incorrect"

    print("✔ Data shapes are correct")


def validate_model(config):
    print("\n🔹 Checking model architecture...")

    model = build_model(
        img_size=tuple(config["model"]["img_size"]),
        num_classes=config["model"]["num_classes"]
    )

    model.summary()

    assert model.input_shape[1:] == tuple(config["model"]["img_size"]) + (3,), \
        "❌ Model input shape wrong"
    assert model.output_shape[-1] == config["model"]["num_classes"], \
        "❌ Model output classes mismatch"

    print("✔ Model architecture is correct")


def validate_single_batch_overfit(config):
    print("\n🔹 Running single-batch overfit test...")

    train_ds = load_dataset(
        directory=config["data"]["train_dir"],
        img_size=tuple(config["model"]["img_size"]),
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    small_ds = train_ds.take(1)

    model = build_model(
        img_size=tuple(config["model"]["img_size"]),
        num_classes=config["model"]["num_classes"]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(small_ds, epochs=15, verbose=0)

    final_acc = history.history["accuracy"][-1]
    print(f"✔ Final single-batch accuracy: {final_acc:.3f}")

    assert final_acc > 0.95, "❌ Model failed to overfit single batch"


def validate_saved_artifacts(config, class_names):
    print("\n🔹 Checking saved artifacts...")

    with open("class_names.json", "r") as f:
        saved_class_names = json.load(f)

    assert saved_class_names == class_names, \
        "❌ class_names.json mismatch"

    print("✔ class_names.json is valid")

    model = tf.keras.models.load_model(
        config["model"]["model_path"]
    )
    print("✔ Model loads successfully")


def validate_inference(config):
    print("\n🔹 Checking inference consistency...")

    model = tf.keras.models.load_model(
        config["model"]["model_path"]
    )

    val_ds = load_dataset(
        directory=config["data"]["val_dir"],
        img_size=tuple(config["model"]["img_size"]),
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    images, labels = next(iter(val_ds))
    preds = model.predict(images, verbose=0)

    pred_labels = np.argmax(preds, axis=1)

    assert preds.shape[0] == images.shape[0], \
        "❌ Prediction batch mismatch"
    assert pred_labels.max() < config["model"]["num_classes"], \
        "❌ Invalid predicted class index"

    print("✔ Inference pipeline is correct")


def main():
    print("\n========== PIPELINE VALIDATION STARTED ==========\n")

    # ---- load config ----
    config = load_config("config/local.yaml")

    # ---- load class names ----
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    validate_data_pipeline(config, class_names)
    validate_model(config)
    validate_single_batch_overfit(config)
    validate_saved_artifacts(config, class_names)
    validate_inference(config)

    print("\n✅ ALL CHECKS PASSED — PIPELINE IS CORRECT 🎉")


if __name__ == "__main__":
    main()
