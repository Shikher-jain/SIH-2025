from pydantic import BaseModel, HttpUrl
from typing import List, Any, Optional

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

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: List[List[float]]
    status: str = "success"
    message: str = "Prediction completed successfully"
    fieldId: Optional[str] = None
    processingTime: Optional[float] = None

class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = "error"
    message: str
    detail: Optional[str] = None