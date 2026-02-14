from pydantic import BaseModel


class PredictionResponse(BaseModel):
    status: str
    predicted_class: str | None = None
    confidence: float | None = None
