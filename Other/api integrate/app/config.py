import os
from pathlib import Path

class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model.h5")
    APP_NAME: str = "CNN Multi-Modal Prediction API"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # File upload limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Model settings
    IMAGE_INPUT_SHAPE: tuple = (224, 224, 3)  # Adjust based on your model
    
settings = Settings()