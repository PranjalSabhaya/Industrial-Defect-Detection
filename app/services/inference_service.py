import numpy as np
from PIL import Image
import io
from src.inference import predict


class InferenceService:

    def __init__(self, model_loader):
        self.model_loader = model_loader

    def predict_image(self, image_bytes: bytes):

        if not image_bytes:
            raise ValueError("Uploaded file is empty")

        # Decode image safely
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        img = np.array(image)

        # Call core inference
        result = predict(
            model=self.model_loader.get_model(),
            config=self.model_loader.get_config(),
            image_array=img
        )

        if result["status"] == "success":
            class_names = self.model_loader.get_class_names()

            result = {
                "status": "success",
                "predicted_class": class_names[result["label"]],
                "confidence": result["confidence"]
            }

        return result
