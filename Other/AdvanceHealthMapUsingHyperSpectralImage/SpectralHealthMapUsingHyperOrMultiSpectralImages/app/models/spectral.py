"""
Spectral analysis database models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class Field(Base):
    """Field information model"""
    __tablename__ = "fields"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(String, unique=True, index=True, nullable=False)
    field_name = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Location and metadata
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    area_hectares = Column(Float, nullable=True)
    crop_type = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    analyses = relationship("SpectralAnalysis", back_populates="field")
    user = relationship("User")


class SpectralAnalysis(Base):
    """Spectral analysis results model"""
    __tablename__ = "spectral_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, unique=True, index=True, nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Analysis metadata
    analysis_type = Column(String, nullable=False)
    processing_status = Column(String, default="pending", nullable=False)
    processing_time_seconds = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Analysis results (stored as JSON)
    vegetation_indices = Column(JSON, nullable=True)
    disease_detection = Column(JSON, nullable=True)
    stress_analysis = Column(JSON, nullable=True)
    anomaly_detection = Column(JSON, nullable=True)
    risk_assessment = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Raw data paths
    hyperspectral_data_path = Column(String, nullable=True)
    multispectral_data_path = Column(String, nullable=True)
    sensor_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    field = relationship("Field", back_populates="analyses")
    user = relationship("User")


class ModelMetrics(Base):
    """Model performance metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    model_version = Column(String, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Training metadata
    training_data_size = Column(Integer, nullable=True)
    training_duration_hours = Column(Float, nullable=True)
    epochs = Column(Integer, nullable=True)
    
    # Additional metrics as JSON
    additional_metrics = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ModelMetrics(model_name='{self.model_name}', version='{self.model_version}', accuracy={self.accuracy})>"


class AnalysisHistory(Base):
    """Historical analysis tracking"""
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Analysis summary
    analysis_date = Column(DateTime(timezone=True), nullable=False)
    health_score = Column(Float, nullable=True)  # Overall health score 0-1
    disease_probability = Column(Float, nullable=True)
    stress_level = Column(Float, nullable=True)
    anomaly_count = Column(Integer, nullable=True)
    
    # Trends
    health_trend = Column(String, nullable=True)  # improving, stable, declining
    risk_level = Column(String, nullable=True)    # low, medium, high, critical
    
    # References
    analysis_id = Column(String, ForeignKey("spectral_analyses.analysis_id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    field = relationship("Field")
    user = relationship("User")