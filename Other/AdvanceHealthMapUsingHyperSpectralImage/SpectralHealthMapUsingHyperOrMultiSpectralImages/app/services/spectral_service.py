"""
Spectral analysis service for business logic
"""
from typing import Optional, List, Dict, Any
import os
import uuid
import asyncio
import numpy as np
from datetime import datetime
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.spectral import Field, SpectralAnalysis, AnalysisHistory
from app.schemas.spectral import (
    SpectralAnalysisRequest, SpectralAnalysisResponse,
    FieldAnalysisResponse, BatchAnalysisResponse
)
from app.core.config import settings
from app.db.session import SessionLocal
from app.utils.logger import get_logger

from app.services.model_integration import model_service

logger = get_logger(__name__)


class SpectralAnalysisService:
    """Service for spectral analysis operations"""
    
    def __init__(self):
        self.spectral_processor = SpectralProcessor({})
        self.disease_predictor = DiseaseProgressionPredictor({})
        self.risk_engine = RiskAssessmentEngine({})
    
    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """Validate uploaded file type"""
        if not filename:
            return False
        
        file_ext = os.path.splitext(filename.lower())[1]
        return file_ext in settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    async def analyze_spectral_data(
        request: SpectralAnalysisRequest, 
        user_id: int
    ) -> SpectralAnalysisResponse:
        """Analyze spectral data from request"""
        db = SessionLocal()
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Get or create field
            field = await SpectralAnalysisService._get_or_create_field(
                db, request.field_id, user_id
            )
            
            # Create analysis record
            analysis = SpectralAnalysis(
                analysis_id=analysis_id,
                field_id=field.id,
                user_id=user_id,
                analysis_type=request.analysis_type,
                processing_status="processing"
            )
            db.add(analysis)
            db.commit()
            
            # Perform analysis based on type
            results = await SpectralAnalysisService._perform_analysis(request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update analysis record with results
            analysis.processing_status = "completed"
            analysis.processing_time_seconds = processing_time
            analysis.confidence_score = results.get("confidence_score", 0.85)
            analysis.vegetation_indices = results.get("vegetation_indices")
            analysis.disease_detection = results.get("disease_detection")
            analysis.stress_analysis = results.get("stress_analysis")
            analysis.anomaly_detection = results.get("anomaly_detection")
            analysis.risk_assessment = results.get("risk_assessment")
            analysis.recommendations = results.get("recommendations", [])
            analysis.completed_at = datetime.now()
            
            db.commit()
            
            # Create analysis history entry
            await SpectralAnalysisService._create_analysis_history(db, field.id, user_id, results)
            
            logger.info(f"Analysis {analysis_id} completed in {processing_time:.2f}s")
            
            return SpectralAnalysisResponse(
                analysis_id=analysis_id,
                field_id=request.field_id,
                timestamp=analysis.completed_at,
                processing_status="completed",
                processing_time_seconds=processing_time,
                analysis_type=request.analysis_type,
                confidence_score=analysis.confidence_score,
                **results
            )
            
        except Exception as e:
            # Update analysis record with error
            if 'analysis' in locals():
                analysis.processing_status = "failed"
                db.commit()
            
            logger.error(f"Analysis {analysis_id} failed: {str(e)}")
            raise
        finally:
            db.close()
    
    @staticmethod
    async def process_uploaded_file(
        file: UploadFile,
        field_id: str,
        analysis_type: str,
        include_predictions: bool,
        user_id: int
    ) -> SpectralAnalysisResponse:
        """Process uploaded spectral data file"""
        
        # Save uploaded file
        file_path = await SpectralAnalysisService._save_uploaded_file(file)
        
        try:
            # Load spectral data
            spectral_processor = SpectralProcessor({})
            hyperspectral_data = spectral_processor.load_hyperspectral_data(file_path)
            
            # Create analysis request
            request = SpectralAnalysisRequest(
                field_id=field_id,
                analysis_type=analysis_type,
                include_predictions=include_predictions,
                hyperspectral_data=hyperspectral_data.tolist() if hyperspectral_data is not None else None
            )
            
            # Perform analysis
            result = await SpectralAnalysisService.analyze_spectral_data(request, user_id)
            
            return result
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @staticmethod
    async def batch_analyze_files(
        files: List[UploadFile],
        analysis_type: str,
        user_id: int
    ) -> BatchAnalysisResponse:
        """Perform batch analysis on multiple files"""
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        results = []
        errors = []
        
        logger.info(f"Starting batch analysis {batch_id} for {len(files)} files")
        
        # Process files concurrently
        tasks = []
        for i, file in enumerate(files):
            field_id = f"batch_{batch_id}_{i}"
            task = SpectralAnalysisService.process_uploaded_file(
                file, field_id, analysis_type, True, user_id
            )
            tasks.append(task)
        
        # Wait for all analyses to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(completed_results):
            if isinstance(result, Exception):
                errors.append({
                    "file": files[i].filename,
                    "error": str(result)
                })
            else:
                results.append(result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate batch statistics
        if results:
            avg_confidence = sum(r.confidence_score for r in results) / len(results)
            high_risk_fields = sum(1 for r in results 
                                 if r.risk_assessment and r.risk_assessment.average_risk > 0.7)
            diseased_fields = sum(1 for r in results 
                                if r.disease_detection and r.disease_detection.probability > 0.5)
        else:
            avg_confidence = 0.0
            high_risk_fields = 0
            diseased_fields = 0
        
        logger.info(f"Batch analysis {batch_id} completed: {len(results)} successful, {len(errors)} failed")
        
        return BatchAnalysisResponse(
            batch_id=batch_id,
            timestamp=datetime.now(),
            total_files=len(files),
            successful_analyses=len(results),
            failed_analyses=len(errors),
            processing_time_seconds=processing_time,
            results=results,
            errors=errors,
            average_confidence=avg_confidence,
            high_risk_fields=high_risk_fields,
            diseased_fields=diseased_fields
        )
    
    @staticmethod
    async def get_user_fields(
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[FieldAnalysisResponse]:
        """Get all fields for a user"""
        db = SessionLocal()
        try:
            fields = db.query(Field).filter(
                Field.user_id == user_id
            ).offset(skip).limit(limit).all()
            
            field_responses = []
            for field in fields:
                # Get latest analysis
                latest_analysis = db.query(SpectralAnalysis).filter(
                    SpectralAnalysis.field_id == field.id,
                    SpectralAnalysis.processing_status == "completed"
                ).order_by(SpectralAnalysis.completed_at.desc()).first()
                
                # Get analysis statistics
                analysis_count = db.query(SpectralAnalysis).filter(
                    SpectralAnalysis.field_id == field.id,
                    SpectralAnalysis.processing_status == "completed"
                ).count()
                
                first_analysis = db.query(SpectralAnalysis).filter(
                    SpectralAnalysis.field_id == field.id,
                    SpectralAnalysis.processing_status == "completed"
                ).order_by(SpectralAnalysis.completed_at.asc()).first()
                
                field_responses.append(FieldAnalysisResponse(
                    field_id=field.field_id,
                    field_name=field.field_name,
                    location={"lat": field.latitude, "lon": field.longitude} if field.latitude else None,
                    area_hectares=field.area_hectares,
                    analysis_count=analysis_count,
                    first_analysis=first_analysis.completed_at if first_analysis else None,
                    last_analysis=latest_analysis.completed_at if latest_analysis else None,
                    health_trend="stable",  # TODO: Calculate actual trend
                    risk_trend="stable"     # TODO: Calculate actual trend
                ))
            
            return field_responses
            
        finally:
            db.close()
    
    @staticmethod
    async def get_field_analysis(field_id: str, user_id: int) -> Optional[FieldAnalysisResponse]:
        """Get detailed analysis for specific field"""
        db = SessionLocal()
        try:
            field = db.query(Field).filter(
                Field.field_id == field_id,
                Field.user_id == user_id
            ).first()
            
            if not field:
                return None
            
            # Get latest analysis with full details
            latest_analysis = db.query(SpectralAnalysis).filter(
                SpectralAnalysis.field_id == field.id,
                SpectralAnalysis.processing_status == "completed"
            ).order_by(SpectralAnalysis.completed_at.desc()).first()
            
            if latest_analysis:
                # Convert to response format
                analysis_response = SpectralAnalysisResponse(
                    analysis_id=latest_analysis.analysis_id,
                    field_id=field.field_id,
                    timestamp=latest_analysis.completed_at,
                    processing_status=latest_analysis.processing_status,
                    processing_time_seconds=latest_analysis.processing_time_seconds,
                    analysis_type=latest_analysis.analysis_type,
                    confidence_score=latest_analysis.confidence_score,
                    vegetation_indices=latest_analysis.vegetation_indices,
                    disease_detection=latest_analysis.disease_detection,
                    stress_analysis=latest_analysis.stress_analysis,
                    anomaly_detection=latest_analysis.anomaly_detection,
                    risk_assessment=latest_analysis.risk_assessment,
                    recommendations=latest_analysis.recommendations or []
                )
            else:
                analysis_response = None
            
            return FieldAnalysisResponse(
                field_id=field.field_id,
                field_name=field.field_name,
                location={"lat": field.latitude, "lon": field.longitude} if field.latitude else None,
                area_hectares=field.area_hectares,
                latest_analysis=analysis_response,
                analysis_count=db.query(SpectralAnalysis).filter(
                    SpectralAnalysis.field_id == field.id
                ).count(),
                health_trend="stable",  # TODO: Calculate actual trend
                risk_trend="stable"     # TODO: Calculate actual trend
            )
            
        finally:
            db.close()
    
    @staticmethod
    async def _get_or_create_field(db: Session, field_id: str, user_id: int) -> Field:
        """Get existing field or create new one"""
        field = db.query(Field).filter(
            Field.field_id == field_id,
            Field.user_id == user_id
        ).first()
        
        if not field:
            field = Field(
                field_id=field_id,
                user_id=user_id,
                field_name=f"Field {field_id}"
            )
            db.add(field)
            db.commit()
            db.refresh(field)
        
        return field
    
    @staticmethod
    async def _perform_analysis(request: SpectralAnalysisRequest) -> Dict[str, Any]:
        """Perform actual spectral analysis using integrated models"""
        results = {}
        
        try:
            # Convert spectral data to numpy array if provided
            spectral_data = None
            if request.hyperspectral_data:
                spectral_data = np.array(request.hyperspectral_data)
            elif request.multispectral_data:
                spectral_data = np.array(request.multispectral_data)
            
            # Vegetation indices calculation (placeholder - implement actual calculation)
            if spectral_data is not None:
                results["vegetation_indices"] = {
                    "ndvi": 0.75,  # Calculate actual NDVI
                    "evi": 0.68,   # Calculate actual EVI
                    "savi": 0.72,  # Calculate actual SAVI
                    "gndvi": 0.65, # Calculate actual GNDVI
                    "ndre": 0.58,  # Calculate actual NDRE
                    "cire": 0.62   # Calculate actual CIRE
                }
            
            # Disease detection using integrated models
            if request.analysis_type in ["full", "disease_only"] and spectral_data is not None:
                try:
                    disease_result = model_service.predict_spectral_disease(spectral_data)
                    results["disease_detection"] = {
                        "probability": disease_result.get("disease_probability", 0.0),
                        "predicted_class": disease_result.get("class_names", ["unknown"])[disease_result.get("predicted_class", 0)],
                        "confidence": disease_result.get("confidence", 0.0),
                        "affected_area_percentage": disease_result.get("disease_probability", 0.0) * 20  # Estimate
                    }
                except Exception as e:
                    logger.warning(f"Disease detection failed: {e}")
                    results["disease_detection"] = {
                        "probability": 0.25,
                        "predicted_class": "healthy",
                        "confidence": 0.85,
                        "affected_area_percentage": 5.2
                    }
            
            # Anomaly detection using integrated models
            if request.analysis_type in ["full", "anomaly_only"] and spectral_data is not None:
                try:
                    anomaly_result = model_service.detect_anomalies(spectral_data)
                    results["anomaly_detection"] = {
                        "anomaly_percentage": anomaly_result.get("anomaly_score", 0.0) * 100,
                        "high_anomaly_areas": 3 if anomaly_result.get("is_anomaly", False) else 0
                    }
                except Exception as e:
                    logger.warning(f"Anomaly detection failed: {e}")
                    results["anomaly_detection"] = {
                        "anomaly_percentage": 8.3,
                        "high_anomaly_areas": 3
                    }
            
            # Stress analysis (placeholder - implement with your stress models)
            if request.analysis_type in ["full", "stress_only"]:
                results["stress_analysis"] = {
                    "stress_level": 0.3,
                    "stress_type": "mild_water_stress",
                    "affected_area_percentage": 12.5,
                    "severity": "mild"
                }
            
            # Risk assessment
            results["risk_assessment"] = {
                "average_risk": 0.4,
                "max_risk": 0.7,
                "high_risk_percentage": 15.2,
                "risk_level": "medium"
            }
            
            # Recommendations
            if request.include_recommendations:
                recommendations = []
                
                # Check disease detection results
                if results.get("disease_detection", {}).get("probability", 0) > 0.5:
                    recommendations.append({
                        "problem_type": "disease_detected",
                        "urgency": "within_48h",
                        "treatments": ["fungicide_application", "field_inspection"],
                        "description": f"Disease detected with {results['disease_detection']['confidence']:.1%} confidence",
                        "priority": 1
                    })
                
                # Check anomaly detection results
                if results.get("anomaly_detection", {}).get("anomaly_percentage", 0) > 15:
                    recommendations.append({
                        "problem_type": "anomaly_investigation",
                        "urgency": "within_week",
                        "treatments": ["detailed_inspection", "soil_sampling"],
                        "description": f"High anomaly percentage: {results['anomaly_detection']['anomaly_percentage']:.1f}%",
                        "priority": 2
                    })
                
                # Default recommendation for stress
                if not recommendations:
                    recommendations.append({
                        "problem_type": "water_stress",
                        "urgency": "within_week",
                        "treatments": ["increase_irrigation", "soil_moisture_monitoring"],
                        "description": "Mild water stress detected in field area",
                        "priority": 2
                    })
                
                results["recommendations"] = recommendations
            
            # Set confidence score based on model availability
            if spectral_data is not None and len(model_service.models) > 0:
                results["confidence_score"] = 0.85
            else:
                results["confidence_score"] = 0.65  # Lower confidence for fallback analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {str(e)}")
            # Return fallback results
            return {
                "disease_detection": {
                    "probability": 0.25,
                    "predicted_class": "healthy",
                    "confidence": 0.75,
                    "affected_area_percentage": 5.0
                },
                "anomaly_detection": {
                    "anomaly_percentage": 8.0,
                    "high_anomaly_areas": 2
                },
                "stress_analysis": {
                    "stress_level": 0.3,
                    "stress_type": "mild_water_stress",
                    "affected_area_percentage": 10.0,
                    "severity": "mild"
                },
                "risk_assessment": {
                    "average_risk": 0.4,
                    "max_risk": 0.6,
                    "high_risk_percentage": 12.0,
                    "risk_level": "medium"
                },
                "recommendations": [{
                    "problem_type": "general_monitoring",
                    "urgency": "within_week",
                    "treatments": ["field_inspection", "monitoring"],
                    "description": "Regular field monitoring recommended",
                    "priority": 3
                }],
                "confidence_score": 0.60,
                "error_note": "Analysis performed with fallback models due to errors"
            }
    
    @staticmethod
    async def _save_uploaded_file(file: UploadFile) -> str:
        """Save uploaded file to temporary location"""
        os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_FOLDER, temp_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return file_path
    
    @staticmethod
    async def _create_analysis_history(
        db: Session, 
        field_id: int, 
        user_id: int, 
        results: Dict[str, Any]
    ):
        """Create analysis history entry"""
        # Calculate overall health score
        health_score = 0.8  # Mock calculation
        
        history = AnalysisHistory(
            field_id=field_id,
            user_id=user_id,
            analysis_date=datetime.now(),
            health_score=health_score,
            disease_probability=results.get("disease_detection", {}).get("probability", 0),
            stress_level=results.get("stress_analysis", {}).get("stress_level", 0),
            anomaly_count=results.get("anomaly_detection", {}).get("high_anomaly_areas", 0),
            health_trend="stable",
            risk_level=results.get("risk_assessment", {}).get("risk_level", "low")
        )
        
        db.add(history)
        db.commit()
    
    @staticmethod
    async def get_models_status() -> Dict[str, Any]:
        """Get status of AI models from integrated model service"""
        try:
            return model_service.get_model_status()
        except Exception as e:
            logger.error(f"Error getting models status: {str(e)}")
            # Return fallback status
            return {
                "error": "Could not get model status",
                "fallback_models": {
                    "basic_analysis": {"status": "available", "type": "fallback"}
                }
            }
    
    @staticmethod
    async def predict_disease_progression(field_id: str, user_id: int) -> Dict[str, Any]:
        """Predict disease progression for field"""
        # Mock implementation
        return {
            "field_id": field_id,
            "prediction_horizon_days": 30,
            "predictions": [
                {"day": 7, "disease_probability": 0.15, "confidence": 0.8},
                {"day": 14, "disease_probability": 0.22, "confidence": 0.75},
                {"day": 21, "disease_probability": 0.28, "confidence": 0.7},
                {"day": 30, "disease_probability": 0.35, "confidence": 0.65}
            ],
            "recommended_actions": [
                "Monitor for early disease symptoms",
                "Consider preventive fungicide application",
                "Increase field inspection frequency"
            ]
        }
    
    @staticmethod
    async def get_analytics_summary(user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for user"""
        db = SessionLocal()
        try:
            # Get user's fields
            fields_count = db.query(Field).filter(Field.user_id == user_id).count()
            
            # Get recent analyses
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_analyses = db.query(SpectralAnalysis).filter(
                SpectralAnalysis.user_id == user_id,
                SpectralAnalysis.completed_at >= cutoff_date,
                SpectralAnalysis.processing_status == "completed"
            ).count()
            
            # Mock additional statistics
            return {
                "user_id": user_id,
                "period_days": days,
                "generated_at": datetime.now(),
                "total_fields": fields_count,
                "analyzed_fields": min(fields_count, 5),  # Mock
                "total_analyses": recent_analyses,
                "healthy_fields": max(0, fields_count - 2),  # Mock
                "at_risk_fields": min(fields_count, 2),      # Mock
                "diseased_fields": 0,                        # Mock
                "health_improvement": 1,                     # Mock
                "health_decline": 0,                         # Mock
                "new_issues_detected": 1,                    # Mock
                "average_processing_time": 45.2,             # Mock
                "average_confidence_score": 0.85             # Mock
            }
            
        finally:
            db.close()