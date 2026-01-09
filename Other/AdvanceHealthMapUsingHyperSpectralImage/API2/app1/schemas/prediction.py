"""
Pydantic schemas for API input and output models
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class SensorData(BaseModel):
    """Schema for environmental sensor data"""
    air_temperature: float = Field(..., description="Air temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture percentage")
    timestamp: str = Field(..., description="Timestamp of sensor reading")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")
    solar_radiation: float = Field(..., ge=0, description="Solar radiation in W/m²")
    leaf_wetness: float = Field(..., ge=0, le=100, description="Leaf wetness percentage")
    co2_level: float = Field(..., ge=0, description="CO2 level in ppm")
    ph_level: float = Field(..., ge=0, le=14, description="Soil pH level")


class ModelInput(BaseModel):
    """Schema for complete model input data"""
    field_id: str = Field(..., description="Unique identifier for the field")
    user_id: str = Field(..., description="Unique identifier for the user")
    hyperspectral_image_url: str = Field(..., description="URL or path to hyperspectral image")
    sensor_data: SensorData = Field(..., description="Environmental sensor data")


class ModelInputFiles(BaseModel):
    """Schema for file-based model input"""
    field_id: str = Field(..., description="Unique identifier for the field")
    user_id: str = Field(..., description="Unique identifier for the user")
    # Files will be handled as UploadFile in the endpoint


class HealthIndices(BaseModel):
    """Schema for vegetation health indices"""
    ndvi: float = Field(..., description="Normalized Difference Vegetation Index")
    evi: float = Field(..., description="Enhanced Vegetation Index")
    savi: float = Field(..., description="Soil Adjusted Vegetation Index")
    ndwi: float = Field(..., description="Normalized Difference Water Index")
    pri: float = Field(..., description="Photochemical Reflectance Index")
    chlorophyll_content: float = Field(..., description="Chlorophyll content estimation")
    leaf_area_index: float = Field(..., description="Leaf Area Index")
    water_stress_index: float = Field(..., description="Water stress indicator")


class CropHealthStatus(BaseModel):
    """Schema for overall crop health status"""
    overall_health_score: float = Field(..., ge=0, le=100, description="Overall health score (0-100)")
    health_category: str = Field(..., description="Health category: Excellent, Good, Fair, Poor, Critical")
    stress_indicators: List[str] = Field(default=[], description="List of detected stress indicators")
    recommendations: List[str] = Field(default=[], description="Management recommendations")
    disease_risk: float = Field(..., ge=0, le=100, description="Disease risk percentage")
    pest_risk: float = Field(..., ge=0, le=100, description="Pest risk percentage")


class ProcessingMetadata(BaseModel):
    """Schema for processing metadata"""
    field_id: str = Field(..., description="Field identifier")
    processing_time: str = Field(..., description="ISO timestamp of processing")
    model_version: str = Field(default="1.0.0", description="Model version used")
    processing_duration: float = Field(..., description="Processing duration in seconds")
    image_resolution: str = Field(..., description="Input image resolution")
    bands_analyzed: int = Field(..., description="Number of spectral bands analyzed")


class ModelOutput(BaseModel):
    """Schema for complete model output"""
    status: str = Field(..., description="Processing status: success, error, processing")
    message: str = Field(..., description="Status message or error description")
    
    # URLs to generated outputs
    health_map_url: Optional[str] = Field(None, description="URL to generated health map image")
    indices_data_url: Optional[str] = Field(None, description="URL to vegetation indices data")
    report_url: Optional[str] = Field(None, description="URL to analysis report PDF")
    sensor_data_url: Optional[str] = Field(None, description="URL to processed sensor data")
    
    # Analysis results
    health_indices: Optional[HealthIndices] = Field(None, description="Calculated vegetation indices")
    crop_health_status: Optional[CropHealthStatus] = Field(None, description="Overall crop health assessment")
    
    # Metadata
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Specific error code")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp")


class HealthStatus(BaseModel):
    """Schema for API health check"""
    status: str = Field(default="healthy", description="API status")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    dependencies: dict = Field(default={}, description="Status of dependencies")