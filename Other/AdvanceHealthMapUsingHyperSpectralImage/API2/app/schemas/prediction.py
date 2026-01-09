from pydantic import BaseModel, HttpUrl
from typing import List, Any, Optional, Dict
from datetime import datetime

class SensorData(BaseModel):
    """Environmental sensor data model."""
    airTemperature: float
    humidity: float
    soilMoisture: float
    timestamp: str
    windSpeed: float
    rainfall: float
    solarRadiation: float
    leafWetness: float
    co2Level: float
    phLevel: float

class PredictionRequest(BaseModel):
    """Request model for field analysis."""
    fieldId: str
    userId: str
    hyperSpectralImageUrl: HttpUrl
    sensorData: SensorData

class ResponseMetadata(BaseModel):
    """Metadata for API response."""
    fieldId: str
    processingTime: str
    modelVersion: str = "2.0.0"
    processingDuration: Optional[float] = None

class PredictionResponse(BaseModel):
    """Response model for field analysis predictions."""
    status: str = "success"
    healthMapUrl: str
    indicesDataUrl: str
    reportUrl: str
    sensorDataUrl: str
    message: str = "Analysis completed successfully"
    metadata: ResponseMetadata

class TrainingRequest(BaseModel):
    """Request model for model training."""
    datasetPath: str
    modelType: str  # "cnn", "autoencoder", "lstm"
    epochs: int = 10
    batchSize: int = 32
    validationSplit: float = 0.2

class TrainingResponse(BaseModel):
    """Response model for training results."""
    status: str
    modelPath: str
    trainingLoss: List[float]
    validationLoss: List[float]
    trainingAccuracy: List[float]
    validationAccuracy: List[float]
    trainingTime: float
    message: str

class ModelStatus(BaseModel):
    """Model status information."""
    modelName: str
    loaded: bool
    modelPath: str
    type: str
    description: str
    lastUpdated: Optional[str] = None

class FieldInfo(BaseModel):
    """Field information model."""
    fieldId: str
    name: str
    location: Dict[str, float]  # {"lat": float, "lon": float}
    area_hectares: float
    crop_type: str
    last_updated: str
    data_available: List[str]

class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = "error"
    message: str
    detail: Optional[str] = None
    error_code: Optional[str] = None