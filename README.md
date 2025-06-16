# Plant Disease Detection Classifier v2

This repository provides a FastAPI service that predicts plant diseases from images using pretrained MobileNetV2 models.

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

