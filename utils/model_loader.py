import os
import json
from keras.models import load_model

models = {}

def get_model(crop_name):
    # app/models 폴더에 있는 모델 파일 경로 계산
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models', f'mobilenetv2_best_{crop_name}.h5')
    )

    if not os.path.exists(model_path):
        print(f"모델 파일 없음: {model_path}")
        return None

    if crop_name not in models:
        print(f"모델 로딩 중: {model_path}")
        models[crop_name] = load_model(model_path)

    return models[crop_name]


def load_label_file(crop_name):
    label_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'mobilenetv2_labels_{crop_name}.json')
    label_path = os.path.abspath(label_path)

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path} 파일이 존재하지 않습니다.")

    with open(label_path, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    return class_mapping