from fastapi import FastAPI, UploadFile, File, HTTPException
from app.core.model_loader import ModelLoader
from app.services.inference_service import InferenceService
from app.schemas import PredictionResponse

app = FastAPI(title="Industrial Defect Detection API")

# ---- Load model at startup ----
model_loader = ModelLoader("config/local.yaml")
model_loader.load()

inference_service = InferenceService(model_loader)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = inference_service.predict_image(image_bytes)

        return PredictionResponse(
            status=result["status"],
            predicted_class=result.get("predicted_class"),
            confidence=result.get("confidence")
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
