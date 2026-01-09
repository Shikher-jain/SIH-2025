"""
Main entry point for the FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.routes import auth, users, spectral_analysis, models
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def create_application() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description="AI-Powered Spectral Health Mapping System API",
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Set up CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(
        auth.router, 
        prefix=f"{settings.API_V1_STR}/auth", 
        tags=["authentication"]
    )
    application.include_router(
        users.router, 
        prefix=f"{settings.API_V1_STR}/users", 
        tags=["users"]
    )
    application.include_router(
        spectral_analysis.router, 
        prefix=f"{settings.API_V1_STR}/spectral", 
        tags=["spectral-analysis"]
    )
    application.include_router(
        models.router, 
        prefix=f"{settings.API_V1_STR}/models", 
        tags=["model-integration"]
    )

    @application.on_event("startup")
    async def startup_event():
        """Initialize application on startup"""
        logger.info("Starting Spectral Health Mapping API")
        
        # Initialize database
        from app.db.init_db import init_db, create_initial_data
        try:
            init_db()
            create_initial_data()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
        
        # Initialize models
        try:
            from app.services.model_integration import model_service
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
        
    @application.on_event("shutdown") 
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("Shutting down Spectral Health Mapping API")

    return application

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )