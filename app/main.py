import logging
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.model_loader import get_model, load_label_file
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
        "apple": "사과",
        "grape": "포도",
        "peach": "복숭아",
        "strawberry" : "딸기"
    }
    cropName_kor = crop_name_map.get(cropName)
    if not cropName_kor:
        return {"error": f"지원하지 않는 작물입니다: {cropName}"}

    model = get_model(cropName_kor)
    if not model:
        return {"error": f"'{cropName_kor}' 모델 파일이 존재하지 않습니다."}

    try:
        class_dict = load_label_file(cropName_kor)  # dict로 로딩
        inv_class_map = {v: k for k, v in class_dict.items()}
    except FileNotFoundError:
        return {"error": f"'{cropName_kor}' 작물의 라벨 파일이 존재하지 않습니다."}

    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 224, 224, 3)

        pred = model.predict(arr)[0]
        predicted_class_idx = int(np.argmax(pred))
        confidence = round(float(np.max(pred)) * 100, 2)

        predicted_class_name = inv_class_map.get(predicted_class_idx, "Unknown")
        sickNameKor = predicted_class_name.split("_")[1] if "_" in predicted_class_name else predicted_class_name

        return {
            "cropName": cropName_kor,
            "predictedClassIndex": predicted_class_idx,
            "sickNameKor": sickNameKor,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"이미지 처리 또는 예측 중 오류 발생: {str(e)}"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)