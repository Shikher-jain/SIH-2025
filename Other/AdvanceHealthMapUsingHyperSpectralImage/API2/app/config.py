import os
from pathlib import Path

class Settings:
    # API Configuration
    APP_NAME: str = "Agricultural ML Analysis API"
    VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/spectral_cnn.h5")
    AUTOENCODER_PATH: str = os.getenv("AUTOENCODER_PATH", "models/spectral_autoencoder.h5")
    LSTM_PATH: str = os.getenv("LSTM_PATH", "models/spectral_lstm.h5")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = ['.npy', '.tif', '.tiff', '.csv', '.jpg', '.png']
    
    # Data Directories
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    SAMPLE_DATA_DIR: str = os.getenv("SAMPLE_DATA_DIR", "data/sample")
    OUTPUTS_DIR: str = os.getenv("OUTPUTS_DIR", "outputs")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    
    # External URLs and Services
    BASE_OUTPUT_URL: str = os.getenv("BASE_OUTPUT_URL", "http://localhost:8000")
    
    # Model Parameters
    IMAGE_INPUT_SHAPE: tuple = (224, 224, 3)
    SENSOR_FEATURES: int = 10
    
    # Processing Configuration
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # Database Configuration (for future use)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./agricultural_data.db")
    
    def __init__(self):
        # Create necessary directories
        for directory in [self.DATA_DIR, self.SAMPLE_DATA_DIR, self.OUTPUTS_DIR, self.MODELS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

settings = Settings()