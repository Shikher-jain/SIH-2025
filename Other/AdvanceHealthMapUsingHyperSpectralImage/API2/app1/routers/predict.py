"""
API Router for prediction endpoints
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import asyncio
import time
import logging
from datetime import datetime
import tempfile
import os
from pathlib import Path

from ..schemas.prediction import (
    ModelInput, ModelInputFiles, ModelOutput, ErrorResponse,
    HealthIndices, CropHealthStatus, ProcessingMetadata
)
from ..models.cnn_model import SpectralCNNModel
from ..utils.preprocessing import HyperspectralProcessor, SensorDataProcessor, DataCombiner
from ..services.output_service import OutputService
from ..core.exceptions import (
    SpectralAPIException, ModelNotLoadedException, InvalidFileFormatException,
    FileTooLargeException, ImageProcessingException, ModelPredictionException
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])

# Global model instance
model_instance = None
output_service = OutputService()


def get_model() -> SpectralCNNModel:
    """Dependency to get model instance"""
    global model_instance
    if model_instance is None:
        model_instance = SpectralCNNModel()
    return model_instance


@router.post("/model-api", response_model=ModelOutput)
async def predict_with_url(
    input_data: ModelInput,
    model: SpectralCNNModel = Depends(get_model)
) -> ModelOutput:
    """
    Predict crop health using hyperspectral image URL and sensor data
    
    Args:
        input_data: Input data containing image URL and sensor data
        model: ML model instance
        
    Returns:
        ModelOutput: Prediction results with generated outputs
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting prediction for field {input_data.field_id}")
        
        # Validate model is loaded
        if not model.is_loaded:
            raise ModelNotLoadedException("ML model is not available")
        
        # Process hyperspectral image
        try:
            # In a real implementation, you would download the image from the URL
            # For now, we'll simulate image processing
            logger.info(f"Processing hyperspectral image: {input_data.hyperspectral_image_url}")
            
            # Simulate image data (replace with actual image loading in production)
            import numpy as np
            image_data = np.random.random((224, 224, 3)).astype(np.float32)
            
        except Exception as e:
            raise ImageProcessingException(f"Failed to process hyperspectral image: {str(e)}")
        
        # Process sensor data
        try:
            sensor_dict = input_data.sensor_data.dict()
            # Convert to numpy array for processing
            sensor_values = [
                sensor_dict['air_temperature'],
                sensor_dict['humidity'],
                sensor_dict['soil_moisture'],
                sensor_dict['wind_speed'],
                sensor_dict['rainfall'],
                sensor_dict['solar_radiation'],
                sensor_dict['leaf_wetness'],
                sensor_dict['co2_level'],
                sensor_dict['ph_level']
            ]
            sensor_data = np.array(sensor_values, dtype=np.float32)
            
            # Normalize sensor data
            sensor_processor = SensorDataProcessor()
            import pandas as pd
            sensor_df = pd.DataFrame([sensor_dict])
            sensor_df = sensor_processor.validate_sensor_ranges(sensor_df)
            sensor_data = sensor_processor.normalize_sensor_data(sensor_df).values[0]
            
        except Exception as e:
            raise ImageProcessingException(f"Failed to process sensor data: {str(e)}")
        
        # Make prediction
        try:
            prediction_results = model.predict(image_data)
        except Exception as e:
            raise ModelPredictionException(f"Model prediction failed: {str(e)}")
        
        # Generate outputs
        processing_time = time.time() - start_time
        
        try:
            # Prepare field info
            field_info = {
                'field_id': input_data.field_id,
                'user_id': input_data.user_id
            }
            
            # Prepare metadata
            metadata_dict = {
                'field_id': input_data.field_id,
                'processing_time': datetime.utcnow().isoformat(),
                'model_version': settings.MODEL_VERSION,
                'processing_duration': processing_time,
                'image_resolution': "224x224",
                'bands_analyzed': 3
            }
            
            # Generate all outputs
            generated_outputs = output_service.generate_all_outputs(
                image_data, prediction_results, field_info, metadata_dict
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate outputs: {str(e)}")
            generated_outputs = {}
        
        # Prepare response
        health_indices = HealthIndices(**prediction_results['vegetation_indices'])
        
        crop_health_status = CropHealthStatus(
            overall_health_score=prediction_results['health_score'],
            health_category=prediction_results['health_category'],
            stress_indicators=prediction_results['stress_indicators'],
            recommendations=prediction_results['recommendations'],
            disease_risk=prediction_results['disease_risk'],
            pest_risk=prediction_results['pest_risk']
        )
        
        metadata = ProcessingMetadata(
            field_id=input_data.field_id,
            processing_time=datetime.utcnow().isoformat(),
            model_version=settings.MODEL_VERSION,
            processing_duration=processing_time,
            image_resolution="224x224",
            bands_analyzed=3
        )
        
        # Generate URLs for outputs
        health_map_url = None
        indices_data_url = None
        report_url = None
        sensor_data_url = None
        
        if generated_outputs:
            if 'health_map' in generated_outputs:
                filename = Path(generated_outputs['health_map']).name
                health_map_url = settings.get_output_url(filename)
            
            if 'indices_json' in generated_outputs:
                filename = Path(generated_outputs['indices_json']).name
                indices_data_url = settings.get_output_url(filename)
            
            if 'report' in generated_outputs:
                filename = Path(generated_outputs['report']).name
                report_url = settings.get_output_url(filename)
            
            if 'indices_csv' in generated_outputs:
                filename = Path(generated_outputs['indices_csv']).name
                sensor_data_url = settings.get_output_url(filename)
        
        response = ModelOutput(
            status="success",
            message="Analysis completed successfully",
            health_map_url=health_map_url,
            indices_data_url=indices_data_url,
            report_url=report_url,
            sensor_data_url=sensor_data_url,
            health_indices=health_indices,
            crop_health_status=crop_health_status,
            metadata=metadata
        )
        
        logger.info(f"Prediction completed for field {input_data.field_id} in {processing_time:.2f}s")
        return response
        
    except SpectralAPIException as e:
        logger.error(f"API error during prediction: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/upload-files", response_model=ModelOutput)
async def predict_with_files(
    field_id: str = Form(...),
    user_id: str = Form(...),
    tif_file: Optional[UploadFile] = File(None),
    npy_file: Optional[UploadFile] = File(None),
    csv_file: Optional[UploadFile] = File(None),
    model: SpectralCNNModel = Depends(get_model)
) -> ModelOutput:
    """
    Predict crop health using uploaded files
    
    Args:
        field_id: Field identifier
        user_id: User identifier
        tif_file: TIF image file
        npy_file: NumPy array file
        csv_file: CSV sensor data file
        model: ML model instance
        
    Returns:
        ModelOutput: Prediction results with generated outputs
    """
    start_time = time.time()
    temp_files = []
    
    try:
        logger.info(f"Starting file-based prediction for field {field_id}")
        
        # Validate model is loaded
        if not model.is_loaded:
            raise ModelNotLoadedException("ML model is not available")
        
        # Validate at least one image file is provided
        if not tif_file and not npy_file:
            raise InvalidFileFormatException("At least one image file (TIF or NPY) must be provided")
        
        # Process image data
        image_data = None
        
        if tif_file:
            # Validate file format and size
            if not tif_file.filename.lower().endswith(('.tif', '.tiff')):
                raise InvalidFileFormatException("TIF file must have .tif or .tiff extension")
            
            if tif_file.size > settings.MAX_FILE_SIZE:
                raise FileTooLargeException(f"TIF file size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes")
            
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
                content = await tif_file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
                
                # Process TIF file
                processor = HyperspectralProcessor()
                image_data = processor.load_tif_image(temp_file.name)
        
        elif npy_file:
            # Validate file format and size
            if not npy_file.filename.lower().endswith('.npy'):
                raise InvalidFileFormatException("NPY file must have .npy extension")
            
            if npy_file.size > settings.MAX_FILE_SIZE:
                raise FileTooLargeException(f"NPY file size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes")
            
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as temp_file:
                content = await npy_file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
                
                # Process NPY file
                processor = HyperspectralProcessor()
                image_data = processor.load_npy_data(temp_file.name)
        
        # Process sensor data
        sensor_data = None
        
        if csv_file:
            # Validate file format and size
            if not csv_file.filename.lower().endswith('.csv'):
                raise InvalidFileFormatException("Sensor data file must have .csv extension")
            
            if csv_file.size > settings.MAX_FILE_SIZE:
                raise FileTooLargeException(f"CSV file size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes")
            
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                content = await csv_file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
                
                # Process CSV file
                processor = SensorDataProcessor()
                sensor_data = processor.load_csv_data(temp_file.name)
        
        # Make prediction
        try:
            prediction_results = model.predict(image_data)
        except Exception as e:
            raise ModelPredictionException(f"Model prediction failed: {str(e)}")
        
        # Generate outputs
        processing_time = time.time() - start_time
        
        try:
            # Prepare field info
            field_info = {
                'field_id': field_id,
                'user_id': user_id
            }
            
            # Prepare metadata
            metadata_dict = {
                'field_id': field_id,
                'processing_time': datetime.utcnow().isoformat(),
                'model_version': settings.MODEL_VERSION,
                'processing_duration': processing_time,
                'image_resolution': f"{image_data.shape[0]}x{image_data.shape[1]}",
                'bands_analyzed': image_data.shape[2] if len(image_data.shape) > 2 else 1
            }
            
            # Generate all outputs
            generated_outputs = output_service.generate_all_outputs(
                image_data, prediction_results, field_info, metadata_dict
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate outputs: {str(e)}")
            generated_outputs = {}
        
        # Prepare response (similar to previous endpoint)
        health_indices = HealthIndices(**prediction_results['vegetation_indices'])
        
        crop_health_status = CropHealthStatus(
            overall_health_score=prediction_results['health_score'],
            health_category=prediction_results['health_category'],
            stress_indicators=prediction_results['stress_indicators'],
            recommendations=prediction_results['recommendations'],
            disease_risk=prediction_results['disease_risk'],
            pest_risk=prediction_results['pest_risk']
        )
        
        metadata = ProcessingMetadata(
            field_id=field_id,
            processing_time=datetime.utcnow().isoformat(),
            model_version=settings.MODEL_VERSION,
            processing_duration=processing_time,
            image_resolution=f"{image_data.shape[0]}x{image_data.shape[1]}",
            bands_analyzed=image_data.shape[2] if len(image_data.shape) > 2 else 1
        )
        
        # Generate URLs for outputs
        health_map_url = None
        indices_data_url = None
        report_url = None
        sensor_data_url = None
        
        if generated_outputs:
            if 'health_map' in generated_outputs:
                filename = Path(generated_outputs['health_map']).name
                health_map_url = settings.get_output_url(filename)
            
            if 'indices_json' in generated_outputs:
                filename = Path(generated_outputs['indices_json']).name
                indices_data_url = settings.get_output_url(filename)
            
            if 'report' in generated_outputs:
                filename = Path(generated_outputs['report']).name
                report_url = settings.get_output_url(filename)
            
            if 'indices_csv' in generated_outputs:
                filename = Path(generated_outputs['indices_csv']).name
                sensor_data_url = settings.get_output_url(filename)
        
        response = ModelOutput(
            status="success",
            message="Analysis completed successfully",
            health_map_url=health_map_url,
            indices_data_url=indices_data_url,
            report_url=report_url,
            sensor_data_url=sensor_data_url,
            health_indices=health_indices,
            crop_health_status=crop_health_status,
            metadata=metadata
        )
        
        logger.info(f"File-based prediction completed for field {field_id} in {processing_time:.2f}s")
        return response
        
    except SpectralAPIException as e:
        logger.error(f"API error during file-based prediction: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during file-based prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")


@router.get("/health")
async def get_model_health(model: SpectralCNNModel = Depends(get_model)) -> Dict[str, Any]:
    """
    Get model health status
    
    Args:
        model: ML model instance
        
    Returns:
        Model health information
    """
    try:
        model_info = model.get_model_info()
        
        return {
            "status": "healthy" if model.is_loaded else "unhealthy",
            "model_loaded": model.is_loaded,
            "model_path": model.model_path,
            "model_version": settings.MODEL_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {
                "tensorflow": "available",
                "numpy": "available",
                "pillow": "available"
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking model health: {str(e)}")
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/info")
async def get_api_info() -> Dict[str, Any]:
    """
    Get API information and capabilities
    
    Returns:
        API information
    """
    return {
        "api_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Spectral Health Mapping API for crop health analysis",
        "endpoints": {
            "/predict/model-api": "Predict using JSON input with image URL and sensor data",
            "/predict/upload-files": "Predict using uploaded files (TIF/NPY images, CSV sensor data)",
            "/predict/health": "Check model health status",
            "/predict/info": "Get API information"
        },
        "supported_formats": {
            "images": list(settings.ALLOWED_IMAGE_EXTENSIONS),
            "data": list(settings.ALLOWED_DATA_EXTENSIONS)
        },
        "max_file_size": settings.MAX_FILE_SIZE,
        "timestamp": datetime.utcnow().isoformat()
    }