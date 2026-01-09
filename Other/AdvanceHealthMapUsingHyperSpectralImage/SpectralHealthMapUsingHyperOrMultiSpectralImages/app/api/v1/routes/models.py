"""
Model integration routes for connecting existing models
"""
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from typing import Dict, Any, List
import numpy as np
from pydantic import BaseModel

from app.core.security import get_current_user, researcher_required
from app.services.model_integration import model_service
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class TextPredictionRequest(BaseModel):
    """Schema for text prediction request"""
    text: str


class SpectralPredictionRequest(BaseModel):
    """Schema for spectral prediction request"""
    spectral_data: List[List[float]]
    model_type: str = "disease"  # disease, anomaly, segmentation


@router.post("/predict/disaster-text")
async def predict_disaster_from_text(
    request: TextPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict disaster information from text using BERT model
    """
    try:
        result = model_service.predict_disaster_text(request.text)
        
        logger.info(f"Disaster text prediction completed for user {current_user['user_id']}")
        
        return {
            "status": "success",
            "input_text": request.text,
            "prediction": result,
            "model_used": "disaster_bert"
        }
        
    except Exception as e:
        logger.error(f"Disaster text prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/spectral-disease")
async def predict_disease_from_spectral(
    request: SpectralPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict disease from spectral data
    """
    try:
        # Convert to numpy array
        spectral_data = np.array(request.spectral_data)
        
        result = model_service.predict_spectral_disease(spectral_data)
        
        logger.info(f"Spectral disease prediction completed for user {current_user['user_id']}")
        
        return {
            "status": "success",
            "data_shape": list(spectral_data.shape),
            "prediction": result,
            "model_used": "spectral_cnn" if "spectral_cnn" in model_service.models else "disease_classifier"
        }
        
    except Exception as e:
        logger.error(f"Spectral disease prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/anomaly-detection")
async def detect_spectral_anomalies(
    request: SpectralPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect anomalies in spectral data
    """
    try:
        # Convert to numpy array
        spectral_data = np.array(request.spectral_data)
        
        result = model_service.detect_anomalies(spectral_data)
        
        logger.info(f"Anomaly detection completed for user {current_user['user_id']}")
        
        return {
            "status": "success",
            "data_shape": list(spectral_data.shape),
            "anomaly_detection": result,
            "model_used": "spectral_autoencoder" if "spectral_autoencoder" in model_service.models else "anomaly_detector"
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.post("/predict/segmentation")
async def segment_spectral_image(
    request: SpectralPredictionRequest,
    current_user: dict = Depends(researcher_required)
):
    """
    Segment spectral image using U-Net (researcher/admin only)
    """
    try:
        # Convert to numpy array
        spectral_data = np.array(request.spectral_data)
        
        result = model_service.segment_spectral_image(spectral_data)
        
        logger.info(f"Spectral segmentation completed for user {current_user['user_id']}")
        
        return {
            "status": "success",
            "data_shape": list(spectral_data.shape),
            "segmentation": result,
            "model_used": "spectral_unet"
        }
        
    except Exception as e:
        logger.error(f"Spectral segmentation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.post("/predict/comprehensive")
async def comprehensive_analysis(
    request: SpectralPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform comprehensive analysis using multiple models
    """
    try:
        # Convert to numpy array
        spectral_data = np.array(request.spectral_data)
        
        results = {}
        
        # Disease prediction
        try:
            results["disease_analysis"] = model_service.predict_spectral_disease(spectral_data)
        except Exception as e:
            results["disease_analysis"] = {"error": str(e)}
        
        # Anomaly detection
        try:
            results["anomaly_analysis"] = model_service.detect_anomalies(spectral_data)
        except Exception as e:
            results["anomaly_analysis"] = {"error": str(e)}
        
        # Segmentation (if user has researcher role)
        user_roles = current_user.get("payload", {}).get("roles", [])
        if any(role in ["admin", "researcher"] for role in user_roles):
            try:
                results["segmentation_analysis"] = model_service.segment_spectral_image(spectral_data)
            except Exception as e:
                results["segmentation_analysis"] = {"error": str(e)}
        
        logger.info(f"Comprehensive analysis completed for user {current_user['user_id']}")
        
        return {
            "status": "success",
            "data_shape": list(spectral_data.shape),
            "comprehensive_results": results,
            "models_used": list(model_service.models.keys())
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive analysis failed: {str(e)}"
        )


@router.get("/models/status")
async def get_models_status(current_user: dict = Depends(get_current_user)):
    """
    Get status of all integrated models
    """
    try:
        status = model_service.get_model_status()
        
        return {
            "status": "success",
            "models": status,
            "total_models": len(status),
            "loaded_models": sum(1 for m in status.values() if m.get("loaded", False))
        }
        
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get models status"
        )


@router.get("/models/health")
async def check_models_health(current_user: dict = Depends(researcher_required)):
    """
    Perform health check on all models (researcher/admin only)
    """
    try:
        health_results = model_service.health_check()
        
        # Calculate overall health
        total_models = len(health_results)
        healthy_models = sum(1 for m in health_results.values() if m.get("status") == "healthy")
        overall_health = "healthy" if healthy_models == total_models else "degraded"
        
        return {
            "overall_health": overall_health,
            "healthy_models": healthy_models,
            "total_models": total_models,
            "model_health": health_results
        }
        
    except Exception as e:
        logger.error(f"Error checking models health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.post("/models/reload")
async def reload_models(current_user: dict = Depends(researcher_required)):
    """
    Reload all models (researcher/admin only)
    """
    try:
        # Reinitialize model service
        model_service._initialize_models()
        
        status = model_service.get_model_status()
        
        logger.info(f"Models reloaded by user {current_user['user_id']}")
        
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "models": status
        }
        
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload models"
        )


@router.post("/predict/batch")
async def batch_prediction(
    files: List[UploadFile] = File(...),
    prediction_type: str = "disease",
    current_user: dict = Depends(researcher_required)
):
    """
    Perform batch predictions on uploaded files (researcher/admin only)
    """
    try:
        results = []
        
        for file in files:
            try:
                # Read file content
                content = await file.read()
                
                # For now, assume files contain spectral data
                # In practice, you'd need proper file parsing based on format
                # This is a simplified example
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": "Batch prediction placeholder - implement file parsing"
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"Batch prediction completed for {len(files)} files by user {current_user['user_id']}")
        
        return {
            "status": "success",
            "total_files": len(files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )