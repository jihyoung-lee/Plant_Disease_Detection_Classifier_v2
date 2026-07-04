import io
from PIL import Image
import numpy as np


def prepare_image(image_bytes: bytes, target=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img