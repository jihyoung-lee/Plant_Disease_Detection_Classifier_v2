import json
import threading
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HFValidationError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


HF_MODEL_REPO = "jihyoung97/plant-disease-models"
HF_MODEL_SUBFOLDER = "models"

models = {}
labels = {}

model_lock = threading.Lock()
label_lock = threading.Lock()


def download_hub_file(filename: str) -> str:
    try:
        return hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            subfolder=HF_MODEL_SUBFOLDER,
        )
    except (
        EntryNotFoundError,
        HFValidationError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    ) as exc:
        raise FileNotFoundError(f"Hugging Face 파일을 불러올 수 없습니다: {filename}") from exc


def get_model(crop_name: str):
    if crop_name in models:
        return models[crop_name]

    with model_lock:
        if crop_name not in models:
            from keras.models import load_model

            model_path = download_hub_file(f"mobilenetv2_best_{crop_name}.h5")
            models[crop_name] = load_model(model_path)

    return models[crop_name]


def load_label_file(crop_name: str):
    if crop_name in labels:
        return labels[crop_name]

    with label_lock:
        if crop_name not in labels:
            label_path = download_hub_file(f"mobilenetv2_labels_{crop_name}.json")
            with open(label_path, "r", encoding="utf-8") as file:
                labels[crop_name] = json.load(file)

    return labels[crop_name]
