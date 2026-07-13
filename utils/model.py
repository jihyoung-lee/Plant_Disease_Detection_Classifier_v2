import numpy as np
from utils.model_loader import get_model

class PlantDiseasePredictor:
    def __init__(self, crop_name, inv_class_map):
        self.crop_name = crop_name
        self.model = get_model(crop_name)
        self.inv_class_map = inv_class_map

    def predict(self, img):
        prob = self.model.predict(img, verbose=0)[0]
        confidence = float(np.max(prob))
        idx = int(np.argmax(prob))

        if confidence < 0.8:
            return "판단보류", round(confidence * 100, 2)

        class_name_full = self.inv_class_map.get(idx, "Unknown")
        sick_name_kor = class_name_full.split("_")[1] if "_" in class_name_full else class_name_full
        return sick_name_kor, round(confidence * 100, 2)
