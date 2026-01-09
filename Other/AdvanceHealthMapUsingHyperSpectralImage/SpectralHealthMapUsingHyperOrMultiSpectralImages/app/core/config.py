"""
Configuration settings for the application
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app settings
    PROJECT_NAME: str = "Spectral Health Mapping API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: Optional[str] = None
    POSTGRES_SERVER: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER: str = "uploads"
    ALLOWED_EXTENSIONS: List[str] = [".tif", ".tiff", ".npy", ".mat", ".npz", ".h5", ".hdf5"]
    
    # AI Model settings
    MODEL_PATH: str = "models/saved"
    ENABLE_GPU: bool = True
    BATCH_SIZE: int = 32
    
    # Spectral data settings
    HYPERSPECTRAL_BANDS: int = 224
    MULTISPECTRAL_BANDS: int = 10
    SPATIAL_RESOLUTION: List[int] = [512, 512]
    
    # Analysis thresholds
    ANOMALY_THRESHOLD: float = 0.7
    DISEASE_THRESHOLD: float = 0.8
    STRESS_THRESHOLD: float = 0.6
    
    # Redis settings (for caching)
    REDIS_URL: Optional[str] = None
    CACHE_EXPIRE: int = 3600  # 1 hour
    
    # Email settings (for notifications)
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()