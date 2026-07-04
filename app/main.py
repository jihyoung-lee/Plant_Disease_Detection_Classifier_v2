import logging

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.service.prediction_service import PredictionService
from app.schemas.prediction import PredictionResponse, PredictionData
from app.exception_handlers import api_exception_handler
from app.exceptions import (
    ApiException,
    EmptyImageException,
    UnsupportedImageTypeException,
)

app = FastAPI()

app.add_exception_handler(
    ApiException,
    api_exception_handler
)

logger = logging.getLogger("uvicorn")

prediction_service = PredictionService()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "병해충 예측 API입니다."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile, crop_name: str = Form(...)):

    if image.content_type not in {"image/jpeg", "image/png"}:
        raise UnsupportedImageTypeException(
            image.content_type or "unknown"
        )
    image_bytes = await image.read()
    if not image_bytes:
        raise EmptyImageException()

    result = prediction_service.predict(image_bytes, crop_name)

    return PredictionResponse(
            success=True,
            message="예측 완료.",
            data=PredictionData(**result),
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
