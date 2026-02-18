import numpy as np
from PIL import Image
import io
from src.inference import predict
from app.core.logger import logger


class InferenceService:

    def __init__(self, model_loader):
        self.model_loader = model_loader

    def predict_image(self, image_bytes: bytes):

        try:
            # ---- Validate empty file ----
            if not image_bytes:
                raise ValueError("Uploaded file is empty")

            logger.info("Received prediction request")

            # ---- Decode safely using PIL ----
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
            img = np.array(image)

            # ---- Validate image resolution ----
            if img.shape[0] < 50 or img.shape[1] < 50:
                raise ValueError("Image resolution too small")

            # ---- Call core inference ----
            result = predict(
                model=self.model_loader.get_model(),
                config=self.model_loader.get_config(),
                image_array=img
            )

            # ---- Handle success case ----
            if result["status"] == "success":
                class_names = self.model_loader.get_class_names()
                predicted_class = class_names[result["label"]]
                confidence = result["confidence"]

                # ---- HANDLE UNKNOWN CLASS ----
                if predicted_class.lower() == "unknown":
                    logger.warning(
                        f"Non-steel image detected | Confidence: {confidence:.6f}"
                    )

                    return {
                        "status": "invalid_input",
                        "message": "Please upload a steel surface defect image.",
                        "confidence": confidence
                    }

                logger.info(
                    f"Prediction successful | "
                    f"Class: {predicted_class} | "
                    f"Confidence: {confidence:.6f}"
                )

                return {
                    "status": "success",
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }

            # ---- Handle uncertain case ----
            else:
                logger.warning(
                    f"Prediction uncertain | "
                    f"Confidence: {result.get('confidence')}"
                )

                return result

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
