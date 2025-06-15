import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from utils.model_loader import get_model

app = FastAPI()
logger = logging.getLogger("uvicorn")

# CORS 설정 (필요 시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로딩
model_cache = {}

@app.get("/")
async def root():
    return {"message": "병해충 예측 API입니다."}
# ====== 예측 엔드포인트 ======
@app.post("/predict")
async def predict(image: UploadFile, cropName: str = Form(...)):
    # 영어 cropName을 한글로 변환
    cropName = cropName.strip()
    crop_name_map = {
        "potato": "감자",
        "tomato": "토마토",
        "apple": "사과"
    }
    cropName_kor = crop_name_map.get(cropName)
    if not cropName_kor:
        return {"error": f"지원하지 않는 작물입니다: {cropName}"}

    model = get_model(cropName_kor)
    class_mappings = {
        "감자": ["겹둥근무늬병", "역병", "건강"]
    }
    class_list = class_mappings.get(cropName_kor)

    if not model:
        return {"error": f"'{cropName_kor}' 모델 파일이 존재하지 않습니다."}
    if not class_list:
        return {"error": f"'{cropName_kor}' {cropName}작물의 클래스 이름이 정의 되어 있지 않습니다."}

    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 224, 224, 3)

        pred = model.predict(arr)[0]
        predicted_class_idx = int(np.argmax(pred))
        confidence = round(float(np.max(pred)) * 100, 2)

        # 인덱스 확인
        if predicted_class_idx >= len(class_list):
            predicted_class_name = "Unknown"
        else:
            predicted_class_name = class_list[predicted_class_idx]

        return {
            "cropName": cropName_kor,
            "predictedClassIndex": predicted_class_idx,
            "sickNameKor": predicted_class_name,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"이미지 처리 또는 예측 중 오류 발생: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)