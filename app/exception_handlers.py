from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions import ApiException

async def api_exception_handler(request: Request, exc: ApiException):
    return JSONResponse(status_code=exc.status_code,
                        content={
                            "success": False,
                            "message": "Request failed.",
                            "data": None,
                            "error": exc.detail,
                        }
                    )