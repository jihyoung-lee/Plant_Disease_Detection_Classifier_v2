import io
from PIL import Image
import numpy as np
from utils.model_loader import get_model

class Predict:
    def __init__(self, crop_name, inv_class_map):
        self.crop_name = crop_name
        self.model = get_model(crop_name)
        self.inv_class_map = inv_class_map

    def prepare_img(self, img_bytes, target=(224, 224)):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(target)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        prob = self.model.predict(img)[0]
        confidence = float(np.max(prob))
        idx = int(np.argmax(prob))

        if confidence < 0.8:
            return "판단불가", round(confidence * 100, 2)

        class_name_full = self.inv_class_map.get(idx, "Unknown")
        sick_name_kor = class_name_full.split("_")[1] if "_" in class_name_full else class_name_full
        return sick_name_kor, round(confidence * 100, 2)
