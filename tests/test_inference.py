import json
from pathlib import Path

from src.utils import load_config
from src.inference import predict


def main():
    # ---- load config ----
    config = load_config("config/local.yaml")

    val_dir = Path(config["data"]["val_dir"])

    # ---- load class names ----
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    # ---- pick one validation image ----
    image_paths = list(val_dir.glob("*/*.jpg"))
    assert len(image_paths) > 0, "No images found in validation directory"

    sample_path = image_paths[0]

    # ---- run inference ----
    result = predict(
        config=config,
        image_path=str(sample_path)
    )

    # ---- handle output ----
    if result["status"] == "uncertain":
        print({
            "status": "uncertain",
            "confidence": round(result["confidence"], 4),
            "image": str(sample_path)
        })

    else:
        print({
            "status": "success",
            "predicted_class": class_names[result["label"]],
            "confidence": round(result["confidence"], 4),
            "image": str(sample_path)
        })


if __name__ == "__main__":
    main()
