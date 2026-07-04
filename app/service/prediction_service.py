from utils.model_loader import load_label_file
from utils.model import PlantDiseasePredictor
from PIL import UnidentifiedImageError
from app.exceptions import (
    UnsupportedCropException,
    LabelFileNotFoundException,
    ModelFileNotFoundException,
    InvalidImageException,
    PredictionFailedException,
)


CROP_NAME_MAP = {
    "potato": "감자",
    "apple": "사과",
    "grape": "포도",
    "peach": "복숭아",
    "strawberry": "딸기",
}

class PredictionService:
    def predict(self,image_bytes: bytes, crop_name: str):

        crop_name = crop_name.strip().lower()
        crop_name_kor = CROP_NAME_MAP.get(crop_name)

        if not crop_name_kor:
            raise UnsupportedCropException(crop_name)

        try:
            label_dict = load_label_file(crop_name_kor)
        except FileNotFoundError as exc:
            raise LabelFileNotFoundException(crop_name_kor) from exc


        inv_class_map = {v: k for k, v in label_dict.items()}

        try:
            predictor = PlantDiseasePredictor(crop_name_kor, inv_class_map)
        except FileNotFoundError as exc:
            raise ModelFileNotFoundException(crop_name_kor) from exc

        try:
            img_array = predictor.prepare_img(image_bytes)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageException() from exc

        try:
            sick_name_kor, confidence = predictor.predict(img_array)
        except Exception as exc:
            raise PredictionFailedException() from exc

        return {
            "crop_name": crop_name_kor,
            "sick_name_kor": sick_name_kor,
            "confidence": confidence,
        }

