"""
Main FastAPI application for Spectral Health Mapping API
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import logging
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from .config import settings
from .routers import predict
from .core.exceptions import SpectralAPIException
from .schemas.prediction import ErrorResponse, HealthStatus

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log") if not settings.DEBUG else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Ensure output directories exist
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if settings.model_exists:
        logger.info(f"Model found at {settings.MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {settings.MODEL_PATH}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced API for crop health analysis using hyperspectral imaging and machine learning",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# Mount static files for serving outputs
if settings.OUTPUT_DIR.exists():
    app.mount("/static/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")


# Custom exception handlers
@app.exception_handler(SpectralAPIException)
async def spectral_api_exception_handler(request: Request, exc: SpectralAPIException):
    """Handle custom API exceptions"""
    logger.error(f"API Exception: {exc.error_code} - {exc.message}")
    
    error_response = ErrorResponse(
        status="error",
        message=exc.message,
        error_code=exc.error_code,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=400,
        content=error_response.dict()
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom format"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_response = ErrorResponse(
        status="error",
        message=str(exc.detail),
        error_code=f"HTTP_{exc.status_code}",
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected Exception: {type(exc).__name__} - {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        status="error",
        message="Internal server error",
        error_code="INTERNAL_SERVER_ERROR",
        details=str(exc) if settings.DEBUG else None,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Include routers
app.include_router(predict.router, prefix=settings.API_PREFIX)


# Root endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "api_prefix": settings.API_PREFIX,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Application health check endpoint"""
    try:
        # Check if model exists
        model_loaded = settings.model_exists
        
        # Check dependencies
        dependencies = {}
        
        try:
            import tensorflow as tf
            dependencies["tensorflow"] = f"available (v{tf.__version__})"
        except ImportError:
            dependencies["tensorflow"] = "not available"
        
        try:
            import numpy as np
            dependencies["numpy"] = f"available (v{np.__version__})"
        except ImportError:
            dependencies["numpy"] = "not available"
        
        try:
            import PIL
            dependencies["pillow"] = f"available (v{PIL.__version__})"
        except ImportError:
            dependencies["pillow"] = "not available"
        
        try:
            import pandas as pd
            dependencies["pandas"] = f"available (v{pd.__version__})"
        except ImportError:
            dependencies["pandas"] = "not available"
        
        # Check directories
        directories_ok = all([
            settings.OUTPUT_DIR.exists(),
            settings.UPLOAD_DIR.exists(),
            settings.TEMP_DIR.exists()
        ])
        
        dependencies["directories"] = "available" if directories_ok else "missing"
        
        # Determine overall status
        status = "healthy"
        if not model_loaded:
            status = "degraded"  # API works but model not available
        
        return HealthStatus(
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            model_loaded=model_loaded,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthStatus(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.APP_VERSION,
            model_loaded=False,
            dependencies={"error": str(e)}
        )


@app.get("/info")
async def get_api_info():
    """Get detailed API information"""
    return {
        "api": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": "Advanced API for crop health analysis using hyperspectral imaging and machine learning",
            "debug_mode": settings.DEBUG
        },
        "endpoints": {
            "root": "/",
            "health": "/health",
            "info": "/info",
            "docs": "/docs",
            "prediction": f"{settings.API_PREFIX}/predict/",
            "model_api": f"{settings.API_PREFIX}/predict/model-api",
            "upload_files": f"{settings.API_PREFIX}/predict/upload-files"
        },
        "configuration": {
            "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
            "supported_image_formats": list(settings.ALLOWED_IMAGE_EXTENSIONS),
            "supported_data_formats": list(settings.ALLOWED_DATA_EXTENSIONS),
            "model_path": settings.MODEL_PATH,
            "model_available": settings.model_exists
        },
        "capabilities": {
            "hyperspectral_analysis": True,
            "vegetation_indices": ["NDVI", "EVI", "SAVI", "NDWI", "PRI"],
            "health_assessment": True,
            "stress_detection": True,
            "risk_analysis": ["disease_risk", "pest_risk"],
            "output_formats": ["health_maps", "analysis_reports", "data_exports"]
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    start_time = datetime.utcnow()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = settings.APP_VERSION
    
    return response


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )