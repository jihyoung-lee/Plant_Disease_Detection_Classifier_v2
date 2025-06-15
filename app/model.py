import io
from PIL import Image
import numpy as np
import tensorflow as tf
from utils.model_loader import get_model

class Predict:
    def __init__(self, crop_name: str, class_names: list):
        self.model = get_model(crop_name)
        self.class_names = class_names

    def prepare_img(self, img_bytes, target=(224, 224)):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(target)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        prob = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()]).predict(img)[0]
        idx = int(np.argmax(prob))
        class_name = self.class_names[idx]
        confidence = round(float(np.max(prob)) * 100, 2)
        return class_name, confidence
