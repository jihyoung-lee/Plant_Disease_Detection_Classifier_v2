from fastapi import FastAPI, UploadFile, Form
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from app.service.prediction_service import PredictionService
from app.schemas.prediction import PredictionResponse, PredictionData
from app.exception_handlers import (
    api_exception_handler,
    global_exception_handler,
    validation_exception_handler,
)
from app.exceptions import (
    ApiException,
    EmptyImageException,
    UnsupportedImageTypeException, ImageTooLargeException,
)

app = FastAPI()

app.add_exception_handler(
    ApiException,
    api_exception_handler
)
app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler
)
app.add_exception_handler(
    Exception,
    global_exception_handler
)

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

    max_file_size = 10 * 1024 * 1024  # 10MB

    image_bytes = await image.read()
    if len(image_bytes) > max_file_size:
        raise ImageTooLargeException()
    if not image_bytes:
        raise EmptyImageException()

    result =  await run_in_threadpool(
    prediction_service.predict,
    image_bytes,
    crop_name,
)
    return PredictionResponse(
            success=True,
            message="예측 완료.",
            data=PredictionData(**result),
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
