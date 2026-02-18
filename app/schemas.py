from typing import Dict, Optional
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    status: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None



class ErrorResponse(BaseModel):
    status: str
    error_code: str
    message: str
