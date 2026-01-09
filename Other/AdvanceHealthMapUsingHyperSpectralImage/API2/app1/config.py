"""
Configuration management for the Spectral Health Mapping API
"""
import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings and configuration"""
    
    # API Configuration
    APP_NAME: str = "Spectral Health Mapping API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/spectral_cnn.h5")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    ALLOWED_IMAGE_EXTENSIONS: set = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    ALLOWED_DATA_EXTENSIONS: set = {".npy", ".csv"}
    
    # Processing Configuration
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Storage Configuration
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", "temp"))
    
    # URLs and Endpoints
    BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")
    STATIC_URL: str = f"{BASE_URL}/static"
    
    # Health Check Configuration
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # 5 minutes
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ALLOWED_METHODS: list = ["GET", "POST", "PUT", "DELETE"]
    ALLOWED_HEADERS: list = ["*"]
    
    def __init__(self):
        # Create necessary directories
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_exists(self) -> bool:
        """Check if the ML model file exists"""
        return Path(self.MODEL_PATH).exists()
    
    def get_output_url(self, filename: str) -> str:
        """Generate URL for output files"""
        return f"{self.STATIC_URL}/outputs/{filename}"
    
    def get_upload_path(self, filename: str) -> Path:
        """Get full path for uploaded files"""
        return self.UPLOAD_DIR / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get full path for output files"""
        return self.OUTPUT_DIR / filename


# Global settings instance
settings = Settings()