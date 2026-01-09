from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with detailed logging."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "timestamp": str(logger.handlers[0].formatter.formatTime(logger.makeRecord("", 0, "", 0, "", (), None))) if logger.handlers else None
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors with enhanced logging."""
    logger.error(f"Unexpected error: {str(exc)} - Path: {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path),
            "error_type": type(exc).__name__
        }
    )

class APIError(Exception):
    """Custom API error class."""
    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)

class ModelNotReadyError(APIError):
    """Error for when ML models are not ready."""
    def __init__(self, model_name: str = "Unknown"):
        super().__init__(
            message=f"Model '{model_name}' is not ready for predictions",
            status_code=503,
            error_code="MODEL_NOT_READY"
        )

class DataProcessingError(APIError):
    """Error for data processing issues."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"Data processing failed: {detail}",
            status_code=400,
            error_code="DATA_PROCESSING_ERROR"
        )