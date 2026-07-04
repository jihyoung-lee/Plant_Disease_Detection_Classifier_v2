from typing import Optional
from pydantic import BaseModel

class PredictionData(BaseModel):
    crop_name: str
    sick_name_kor: str
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[PredictionData] = None