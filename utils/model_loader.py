import os
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
