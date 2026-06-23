import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.model_loader import load_label_file
from utils.model import Predict

app = FastAPI()
logger = logging.getLogger("uvicorn")

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

@app.post("/predict")
async def predict(image: UploadFile, crop_name: str = Form(...)):
    crop_name = crop_name.strip()

    crop_name_map = {
        "potato": "감자",
        "tomato": "토마토",
        "apple": "사과",
        "grape": "포도",
        "peach": "복숭아",
        "strawberry": "딸기"
    }

    crop_name_kor = crop_name_map.get(crop_name)
    if not crop_name_kor:
        return {"error": f"지원하지 않는 작물입니다: {crop_name}"}

    try:
        label_dict = load_label_file(crop_name_kor)
        inv_class_map = {v: k for k, v in label_dict.items()}
        predictor = Predict(crop_name_kor, inv_class_map)

        img_bytes = await image.read()
        img_array = predictor.prepare_img(img_bytes)
        class_name, confidence = predictor.predict(img_array)

        return {
            "cropName": crop_name_kor,
            "sickNameKor": class_name,
            "confidence": confidence
        }

    except FileNotFoundError:
        return {"error": f"'{crop_name_kor}' 작물의 라벨 파일이 존재하지 않습니다."}
    except Exception as e:
        return {"error": f"이미지 처리 또는 예측 중 오류 발생: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
