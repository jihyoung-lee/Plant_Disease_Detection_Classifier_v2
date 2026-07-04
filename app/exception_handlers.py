import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.exceptions import ApiException

logger = logging.getLogger("uvicorn")

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = exc.errors()

    first_error = errors[0] if errors else {}
    field = first_error.get("loc", ["unknown"])[-1]
    message = first_error.get("msg", "Validation failed.")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation failed.",
            "data": None,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"{field}: {message}",
                "details": errors,
            },
        },
    )

async def api_exception_handler(
        request: Request,
        exc: ApiException,
):
    return JSONResponse(status_code=exc.status_code,
                        content={
                            "success": False,
                            "message": "Request failed.",
                            "data": None,
                            "error": exc.detail,
                        }
                    )

async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(exc)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal Server Error",
            "data": None,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "예상하지 못한 오류가 발생했습니다."
            }
        }
    )