from fastapi import HTTPException

class ApiException(HTTPException):

    def __init__(self, status_code: int, code: str, detail: str):
        super().__init__(
            status_code=status_code,
            detail={
                "code": code,
                "message": detail
            }
        )

class UnsupportedCropException(ApiException):
    def __init__(self, crop_name: str):
        super().__init__(
            status_code=400,
            code="UNSUPPORTED_CROP",
            detail=f"지원하지 않는 작물입니다: {crop_name}"
        )

class ModelFileNotFoundException(ApiException):
    def __init__(self, crop_name: str):
        super().__init__(
            status_code=500,
            code="MODEL_FILE_NOT_FOUND",
            detail=f"{crop_name} 모델 파일을 찾을 수 없습니다."
        )

class LabelFileNotFoundException(ApiException):
    def __init__(self, crop_name: str):
        super().__init__(
            status_code=500,
            code="LABEL_FILE_NOT_FOUND",
            detail=f"{crop_name} 라벨 파일을 찾을 수 없습니다."
        )

class PredictionFailedException(ApiException):
    def __init__(self):
        super().__init__(
            status_code=500,
            code="PREDICTION_FAILED",
            detail="예측에 실패했습니다."
        )

class EmptyImageException(ApiException):
    def __init__(self):
        super().__init__(
            status_code=400,
            code="EMPTY_IMAGE",
            detail="이미지 파일이 비어 있습니다.",
        )


class UnsupportedImageTypeException(ApiException):
    def __init__(self, content_type: str):
        super().__init__(
            status_code=415,
            code="UNSUPPORTED_IMAGE_TYPE",
            detail=f"지원하지 않는 이미지 형식입니다: {content_type}",
        )


class InvalidImageException(ApiException):
    def __init__(self):
        super().__init__(
            status_code=422,
            code="INVALID_IMAGE",
            detail="이미지 파일을 해석할 수 없습니다.",
        )


class ModelLoadFailedException(ApiException):
    def __init__(self, crop_name: str):
        super().__init__(
            status_code=500,
            code="MODEL_LOAD_FAILED",
            detail=f"{crop_name} 모델을 불러오지 못했습니다.",
        )

class ImageTooLargeException(ApiException):
    def __init__(self):
        super().__init__(
            status_code=413,
            code="FILE_TOO_LARGE",
            detail="이미지 파일 용량 초과"
        )