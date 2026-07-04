from utils.model_loader import load_label_file
from utils.model import PlantDiseasePredictor


CROP_NAME_MAP = {
    "potato": "감자",
    "apple": "사과",
    "grape": "포도",
    "peach": "복숭아",
    "strawberry": "딸기",
}

class PredictionService:
    def predict(self,image_bytes: bytes, crop_name: str):

        crop_name = crop_name.strip()
        crop_name_kor = CROP_NAME_MAP.get(crop_name)

        if not crop_name_kor:
            raise ValueError(f"지원하지 않는 작물입니다: {crop_name}")


        label_dict = load_label_file(crop_name_kor)
        inv_class_map = {v: k for k, v in label_dict.items()}

        predictor = PlantDiseasePredictor(crop_name_kor, inv_class_map)

        img_array = predictor.prepare_img(image_bytes)
        sick_name_kor, confidence = predictor.predict(img_array)

        return {
            "crop_name": crop_name_kor,
            "sick_name_kor": sick_name_kor,
            "confidence": confidence,
        }
    
