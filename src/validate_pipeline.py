import json
import tensorflow as tf
import numpy as np
from src.data_pipeline import load_dataset
from src.model import build_model
from src.config import TRAIN_DIR, VAL_DIR, IMG_SIZE, NUM_CLASSES, MODEL_PATH


def validate_data_pipeline():
    print("🔹 Checking data pipeline...")

    train_ds,train_classes = load_dataset(TRAIN_DIR)
    val_ds,val_classes = load_dataset(VAL_DIR, shuffle=False)

    # class names consistency
    assert train_classes == val_classes, \
        "❌ Train and Val class names mismatch"

    print("✔ Class names:", train_classes)

    # batch shape check
    for images, labels in train_ds.take(1):
        assert images.shape[1:] == IMG_SIZE + (3,), "❌ Image shape mismatch"
        assert len(labels.shape) == 1, "❌ Labels shape incorrect"

    print("✔ Data shapes are correct")


def validate_model():
    print("\n🔹 Checking model architecture...")

    model = build_model()
    model.summary()

    assert model.input_shape[1:] == IMG_SIZE + (3,), "❌ Model input shape wrong"
    assert model.output_shape[-1] == NUM_CLASSES, "❌ Model output classes mismatch"

    print("✔ Model architecture is correct")
    return model


def validate_single_batch_overfit(model):
    print("\n🔹 Running single-batch overfit test...")

    train_ds,class_names = load_dataset(TRAIN_DIR)
    small_ds = train_ds.take(1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(small_ds, epochs=15, verbose=0)

    final_acc = history.history["accuracy"][-1]
    print(f"✔ Final single-batch accuracy: {final_acc:.3f}")

    assert final_acc > 0.95, "❌ Model failed to overfit single batch"


def validate_saved_artifacts():
    print("\n🔹 Checking saved artifacts...")

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    train_ds,train_classes = load_dataset(TRAIN_DIR)

    assert class_names == train_classes, \
        "❌ Saved class_names.json does not match training dataset"

    print("✔ class_names.json is valid")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("✔ Model loads successfully")


def validate_inference():
    print("\n🔹 Checking inference consistency...")

    model = tf.keras.models.load_model(MODEL_PATH)
    val_ds,_ = load_dataset(VAL_DIR, shuffle=False)

    images, labels = next(iter(val_ds))
    preds = model.predict(images, verbose=0)

    pred_labels = np.argmax(preds, axis=1)

    assert preds.shape[0] == images.shape[0], "❌ Prediction batch mismatch"
    assert pred_labels.max() < NUM_CLASSES, "❌ Invalid predicted class index"

    print("✔ Inference pipeline is correct")


def main():
    print("\n========== PIPELINE VALIDATION STARTED ==========\n")

    validate_data_pipeline()
    model = validate_model()
    validate_single_batch_overfit(model)
    validate_saved_artifacts()
    validate_inference()

    print("\n✅ ALL CHECKS PASSED — PIPELINE IS CORRECT 🎉")


if __name__ == "__main__":
    main()
