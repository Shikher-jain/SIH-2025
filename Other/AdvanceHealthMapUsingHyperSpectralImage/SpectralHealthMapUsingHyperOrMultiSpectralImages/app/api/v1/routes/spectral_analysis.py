"""
Spectral analysis routes
"""
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from typing import List, Optional
import asyncio
from app.core.security import get_current_user, researcher_required
from app.schemas.spectral import (
    SpectralAnalysisRequest, SpectralAnalysisResponse, 
    FieldAnalysisResponse, BatchAnalysisResponse
)
from app.services.spectral_service import SpectralAnalysisService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/analyze", response_model=SpectralAnalysisResponse)
async def analyze_spectral_data(
    request: SpectralAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze uploaded spectral data
    """
    try:
        # Perform spectral analysis
        result = await SpectralAnalysisService.analyze_spectral_data(
            request, current_user["user_id"]
        )
        
        logger.info(f"Spectral analysis completed for user {current_user['user_id']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Spectral analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spectral analysis failed"
        )


@router.post("/upload-analyze", response_model=SpectralAnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    field_id: str = Form(...),
    analysis_type: str = Form("full"),
    include_predictions: bool = Form(True),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload spectral data file and perform analysis
    """
    try:
        # Validate file type
        if not SpectralAnalysisService.validate_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Supported: .tif, .tiff, .npy, .mat, .npz, .h5, .hdf5"
            )
        
        # Process uploaded file
        result = await SpectralAnalysisService.process_uploaded_file(
            file, field_id, analysis_type, include_predictions, current_user["user_id"]
        )
        
        logger.info(f"File {file.filename} analyzed for user {current_user['user_id']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and analyze error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File analysis failed"
        )


@router.get("/fields", response_model=List[FieldAnalysisResponse])
async def list_analyzed_fields(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """
    List all analyzed fields for current user
    """
    try:
        fields = await SpectralAnalysisService.get_user_fields(
            current_user["user_id"], skip=skip, limit=limit
        )
        
        return fields
        
    except Exception as e:
        logger.error(f"Error listing fields: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list fields"
        )


@router.get("/fields/{field_id}", response_model=FieldAnalysisResponse)
async def get_field_analysis(
    field_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed analysis for specific field
    """
    try:
        field_analysis = await SpectralAnalysisService.get_field_analysis(
            field_id, current_user["user_id"]
        )
        
        if not field_analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Field analysis not found"
            )
        
        return field_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get field analysis"
        )


@router.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze(
    files: List[UploadFile] = File(...),
    analysis_type: str = Form("full"),
    current_user: dict = Depends(researcher_required)
):
    """
    Perform batch analysis on multiple files (researcher/admin only)
    """
    try:
        # Validate all files
        for file in files:
            if not SpectralAnalysisService.validate_file_type(file.filename):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {file.filename}"
                )
        
        # Process files in batch
        results = await SpectralAnalysisService.batch_analyze_files(
            files, analysis_type, current_user["user_id"]
        )
        
        logger.info(f"Batch analysis completed for {len(files)} files by user {current_user['user_id']}")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch analysis failed"
        )


@router.get("/models/status")
async def get_models_status(current_user: dict = Depends(get_current_user)):
    """
    Get status of AI models
    """
    try:
        status_info = await SpectralAnalysisService.get_models_status()
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get models status"
        )


@router.post("/predictions/disease")
async def predict_disease(
    field_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate disease predictions for specific field
    """
    try:
        predictions = await SpectralAnalysisService.predict_disease_progression(
            field_id, current_user["user_id"]
        )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Disease prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Disease prediction failed"
        )


@router.get("/analytics/summary")
async def get_analytics_summary(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """
    Get analytics summary for user's fields
    """
    try:
        summary = await SpectralAnalysisService.get_analytics_summary(
            current_user["user_id"], days=days
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Analytics summary error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics summary"
        )