from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.model_loader import ModelLoader
from app.services.inference_service import InferenceService
from app.schemas import PredictionResponse, ErrorResponse
from app.core.logger import logger

app = FastAPI(title="Industrial Defect Detection API")

# ---- Load model at startup ----
model_loader = ModelLoader("config/local.yaml")
model_loader.load()

inference_service = InferenceService(model_loader)


# -------------------------
# Global Exception Handler
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Something went wrong"
        }
    )


# -------------------------
# Health Check
# -------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse},
               415: {"model": ErrorResponse}}
)
async def predict_image(file: UploadFile = File(...)):

    # ---- Validate file type ----
    if not file.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded")

        raise HTTPException(
            status_code=415,
            detail={
                "status": "error",
                "error_code": "INVALID_FILE_TYPE",
                "message": "Only image files are allowed"
            }
        )

    try:
        image_bytes = await file.read()

        result = inference_service.predict_image(image_bytes)

        return result

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")

        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "error_code": "INVALID_IMAGE",
                "message": str(e)
            }
        )
