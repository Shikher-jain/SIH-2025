from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )