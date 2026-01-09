"""
Spectral analysis schemas for request/response validation
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, validator
from datetime import datetime
import numpy as np


class SpectralAnalysisRequest(BaseModel):
    """Schema for spectral analysis request"""
    field_id: str
    analysis_type: str = "full"  # full, quick, disease_only, stress_only
    include_predictions: bool = True
    include_recommendations: bool = True
    hyperspectral_data: Optional[List[List[float]]] = None
    multispectral_data: Optional[List[List[float]]] = None
    sensor_data: Optional[Dict[str, float]] = None
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = ['full', 'quick', 'disease_only', 'stress_only', 'anomaly_only']
        if v not in allowed_types:
            raise ValueError(f'Invalid analysis type: {v}')
        return v


class VegetationIndices(BaseModel):
    """Vegetation indices schema"""
    ndvi: float
    evi: float
    savi: float
    gndvi: float
    ndre: float
    cire: float


class DiseaseDetection(BaseModel):
    """Disease detection results schema"""
    probability: float
    predicted_class: str
    confidence: float
    affected_area_percentage: float
    disease_type: Optional[str] = None


class StressAnalysis(BaseModel):
    """Stress analysis results schema"""
    stress_level: float
    stress_type: str
    affected_area_percentage: float
    severity: str  # mild, moderate, severe


class AnomalyDetection(BaseModel):
    """Anomaly detection results schema"""
    anomaly_percentage: float
    high_anomaly_areas: int
    anomaly_map: Optional[List[List[float]]] = None


class RiskAssessment(BaseModel):
    """Risk assessment schema"""
    average_risk: float
    max_risk: float
    high_risk_percentage: float
    risk_level: str  # low, medium, high, critical


class Recommendation(BaseModel):
    """Recommendation schema"""
    problem_type: str
    urgency: str
    treatments: List[str]
    description: str
    priority: int = 1


class SpectralAnalysisResponse(BaseModel):
    """Schema for spectral analysis response"""
    analysis_id: str
    field_id: str
    timestamp: datetime
    processing_status: str
    
    # Analysis results
    vegetation_indices: Optional[VegetationIndices] = None
    disease_detection: Optional[DiseaseDetection] = None
    stress_analysis: Optional[StressAnalysis] = None
    anomaly_detection: Optional[AnomalyDetection] = None
    risk_assessment: Optional[RiskAssessment] = None
    
    # Recommendations
    recommendations: List[Recommendation] = []
    
    # Metadata
    processing_time_seconds: float
    analysis_type: str
    confidence_score: float


class FieldAnalysisResponse(BaseModel):
    """Schema for field analysis response"""
    field_id: str
    field_name: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # lat, lon
    area_hectares: Optional[float] = None
    
    # Latest analysis
    latest_analysis: Optional[SpectralAnalysisResponse] = None
    
    # Historical data
    analysis_count: int
    first_analysis: Optional[datetime] = None
    last_analysis: Optional[datetime] = None
    
    # Trends
    health_trend: str  # improving, stable, declining
    risk_trend: str    # increasing, stable, decreasing


class BatchAnalysisResponse(BaseModel):
    """Schema for batch analysis response"""
    batch_id: str
    timestamp: datetime
    total_files: int
    successful_analyses: int
    failed_analyses: int
    processing_time_seconds: float
    
    # Results summary
    results: List[SpectralAnalysisResponse]
    errors: List[Dict[str, str]] = []
    
    # Batch statistics
    average_confidence: float
    high_risk_fields: int
    diseased_fields: int


class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    field_id: str
    prediction_type: str  # disease_progression, yield_forecast, risk_evolution
    prediction_horizon_days: int = 30
    include_confidence_intervals: bool = True


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    field_id: str
    prediction_type: str
    prediction_horizon_days: int
    generated_at: datetime
    
    # Predictions
    predictions: List[Dict[str, Any]]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    
    # Model info
    model_version: str
    accuracy_score: float


class AnalyticsSummary(BaseModel):
    """Schema for analytics summary"""
    user_id: int
    period_days: int
    generated_at: datetime
    
    # Field statistics
    total_fields: int
    analyzed_fields: int
    total_analyses: int
    
    # Health metrics
    healthy_fields: int
    at_risk_fields: int
    diseased_fields: int
    
    # Trends
    health_improvement: int
    health_decline: int
    new_issues_detected: int
    
    # Performance metrics
    average_processing_time: float
    average_confidence_score: float