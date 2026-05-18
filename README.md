# Plant Disease Detection Classifier v2

This repository provides a FastAPI service that predicts plant diseases from images using pretrained MobileNetV2 models.

## Dataset & Model
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

<img width="700" height="448" alt="image" src="https://github.com/user-attachments/assets/eb175dea-949f-4e94-b380-c10c67206513" />

A MobileNetV2 pretrained backbone was used as a baseline model for the Apple(사과) disease classification task, achieving approximately 98% validation accuracy.
However, since the PlantVillage dataset contains many images captured under controlled backgrounds and lighting conditions, there may be limitations in generalizing to real-world field environments.

<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/51e9a42d-0112-4e2d-b715-4f8aa07b3c1f" />

The confusion matrix shows that the model correctly classified most Apple disease categories with minimal confusion between classes.
Minor misclassifications occurred mainly in the Apple Scab category, which may be caused by visual similarities in leaf lesion patterns or variations in image conditions.
## Features

- FastAPI-based HTTP API
- Pretrained Keras models located in `models/`
- Supports crops such as potato, tomato, apple, grape, peach and strawberry (in Korean labels)

## Requirements

- Python 3.8+
- fastapi
- uvicorn
- keras
- pillow
- numpy

## Running the API

```bash
uvicorn app.main:app --reload
```

Then access `http://localhost:8000` or use the `/predict` endpoint with an image file and crop name.

