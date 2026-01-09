from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict
from app.config import settings
from app.core.exceptions import http_exception_handler, general_exception_handler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="CNN-based multi-modal prediction API for TIF images, NPY arrays, and CSV sensor data",
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(predict.router)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/predict/health",
        "api_status": "/predict/status",
        "endpoints": {
            "predict_json": "/predict/",
            "predict_files": "/predict/files"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info("Model loading completed during import")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.DEBUG
    )