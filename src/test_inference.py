import json
from src.inference import predict
from src.config import VAL_DIR

# load class names
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# get one sample image safely
image_paths = list(VAL_DIR.glob("*/*.jpg"))
assert len(image_paths) > 0, "No images found in validation directory"

sample_path = image_paths[0]

# run inference
label, confidence = predict(str(sample_path))

# handle low confidence
if confidence < 0.6:
    print({
        "status": "uncertain",
        "confidence": round(confidence, 4)
    })
else:
    print({
        "status": "success",
        "predicted_class": CLASS_NAMES[label],
        "confidence": round(confidence, 4),
        "image": str(sample_path)
    })
