import logging

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.service.prediction_service import PredictionService
from app.schemas.prediction import PredictionResponse, PredictionData

app = FastAPI()
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

    try:
        image_bytes = await image.read()
        result = prediction_service.predict(image_bytes, crop_name)

        return PredictionResponse(
            success=True,
            message="예측 완료.",
            data=PredictionData(**result),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        logger.exception("모델이나 해당 라벨을 찾을 수 없습니다.")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="이미지 처리 또는 예측 중 오류가 발생했습니다."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
