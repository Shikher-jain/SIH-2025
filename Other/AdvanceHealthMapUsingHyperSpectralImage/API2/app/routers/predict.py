from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.models.cnn_model import ml_models
from app.utils.preprocessing import preprocessor
from app.schemas.prediction import (
    PredictionResponse, ErrorResponse, PredictionRequest, 
    ModelStatus, FieldInfo, TrainingRequest, TrainingResponse
)
from app.services.output_service import output_service
from app.core.exceptions import ModelNotReadyError, DataProcessingError
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Agricultural Analysis"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)

@router.post("/model-api/", response_model=PredictionResponse)
async def model_api_endpoint(request_data: PredictionRequest):
    """
    Main model API endpoint for comprehensive agricultural field analysis.
    
    This endpoint processes agricultural field data including:
    - Field identification and user context
    - Hyperspectral image data (downloaded from URL)
    - Environmental sensor readings
    
    Returns comprehensive analysis with:
    - Health map visualization URL
    - Vegetation indices data URL
    - Detailed analysis report URL
    - Processed sensor data URL
    - Processing metadata
    
    Example request:
    {
        "fieldId": "field_123",
        "userId": "user_456",
        "hyperSpectralImageUrl": "http://example.com/image.npy",
        "sensorData": {
            "airTemperature": 25.5,
            "humidity": 60,
            "soilMoisture": 30,
            "timestamp": "2025-10-03T10:00:00Z",
            "windSpeed": 10,
            "rainfall": 0,
            "solarRadiation": 800,
            "leafWetness": 15,
            "co2Level": 400,
            "phLevel": 6.5
        }
    }
    """
    processing_start_time = datetime.now()
    
    try:
        logger.info(f"Starting comprehensive analysis for field {request_data.fieldId}")
        
        # Ensure models are loaded
        ml_models.ensure_models_loaded()
        
        if not ml_models.is_ready():
            raise ModelNotReadyError("CNN")
        
        # Download and process hyperspectral image
        logger.info(f"Downloading hyperspectral image from {request_data.hyperSpectralImageUrl}")
        hyperspectral_data = preprocessor.download_hyperspectral_image(str(request_data.hyperSpectralImageUrl))
        
        # Process sensor data
        logger.info("Processing environmental sensor data")
        sensor_array = preprocessor.process_sensor_data(request_data.sensorData)
        
        # Combine data for model input
        logger.info("Combining multimodal data for CNN model")
        combined_data = preprocessor.combine_multimodal_data(hyperspectral_data, sensor_array)
        
        # Make CNN prediction
        logger.info("Making prediction with CNN model")
        prediction = ml_models.predict_cnn(combined_data)
        
        # Extract features using autoencoder if available
        if ml_models._models_loaded['autoencoder']:
            logger.info("Extracting features with autoencoder")
            features = ml_models.extract_features(hyperspectral_data.reshape(1, *hyperspectral_data.shape))
        
        # Generate comprehensive outputs
        logger.info("Generating comprehensive analysis outputs")
        
        # Generate advanced health map with hyperspectral data
        health_map_url = output_service.generate_health_map(
            prediction, request_data.fieldId, hyperspectral_data
        )
        
        # Generate detailed indices data
        indices_data_url = output_service.generate_indices_data(
            prediction, request_data.fieldId, hyperspectral_data
        )
        
        # Generate comprehensive analysis report
        sensor_dict = request_data.sensorData.dict()
        report_url = output_service.generate_analysis_report(
            prediction, request_data.fieldId, sensor_dict
        )
        
        # Save enhanced sensor data
        sensor_data_url = output_service.save_sensor_data(sensor_dict, request_data.fieldId)
        
        # Create response metadata with processing time
        metadata = output_service.create_response_metadata(
            request_data.fieldId, processing_start_time
        )
        
        logger.info(f"Analysis completed successfully for field {request_data.fieldId}")
        
        return PredictionResponse(
            healthMapUrl=health_map_url,
            indicesDataUrl=indices_data_url,
            reportUrl=report_url,
            sensorDataUrl=sensor_data_url,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except ModelNotReadyError as e:
        logger.error(f"Model not ready: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except DataProcessingError as e:
        logger.error(f"Data processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during analysis: {str(e)}"
        )

@router.post("/analyze", response_model=PredictionResponse)
async def analyze_field_data(request_data: PredictionRequest):
    """Alias endpoint for analyze_field_data - same as model-api/"""
    return await model_api_endpoint(request_data)

@router.post("/upload", response_model=PredictionResponse)
async def upload_multimodal_files(
    tif_file: UploadFile = File(..., description="TIF hyperspectral image file"),
    npy_file: UploadFile = File(..., description="NumPy array file (.npy)"),
    csv_file: UploadFile = File(..., description="CSV sensor data file"),
):
    """
    Legacy multi-modal prediction endpoint that accepts file uploads.
    
    - **tif_file**: Upload a TIF hyperspectral image file
    - **npy_file**: Upload a NumPy array file (.npy format)
    - **csv_file**: Upload a CSV file containing sensor data
    
    Returns comprehensive prediction results using the advanced CNN model.
    """
    processing_start_time = datetime.now()
    
    try:
        # Validate file types
        if not tif_file.filename.lower().endswith(('.tif', '.tiff')):
            raise HTTPException(status_code=400, detail="TIF file must have .tif or .tiff extension")
        
        if not npy_file.filename.lower().endswith('.npy'):
            raise HTTPException(status_code=400, detail="NumPy file must have .npy extension")
            
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Sensor data file must have .csv extension")
        
        # Ensure models are loaded
        ml_models.ensure_models_loaded()
        
        if not ml_models.is_ready():
            raise ModelNotReadyError("CNN")
        
        # Process each file
        logger.info("Processing uploaded files...")
        
        # Load and preprocess data
        image_data = preprocessor.load_tif_image(tif_file)
        npy_data = preprocessor.load_npy_file(npy_file)
        sensor_data = preprocessor.load_csv_file(csv_file)
        
        # Combine data for model input (legacy method)
        combined_data = preprocessor.combine_legacy_data(image_data, npy_data, sensor_data)
        
        # Make prediction
        logger.info("Making prediction with advanced CNN model...")
        prediction = ml_models.predict_cnn(combined_data)
        
        # Generate field ID for legacy uploads
        field_id = f"upload_{hash(tif_file.filename)}_{hash(npy_file.filename)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Generate comprehensive outputs
        health_map_url = output_service.generate_health_map(prediction, field_id, image_data)
        indices_data_url = output_service.generate_indices_data(prediction, field_id, image_data)
        
        # Create sensor data dict for report
        sensor_dict = {
            "upload_method": "legacy_files",
            "tif_filename": tif_file.filename,
            "npy_filename": npy_file.filename,
            "csv_filename": csv_file.filename,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        report_url = output_service.generate_analysis_report(prediction, field_id, sensor_dict)
        sensor_data_url = output_service.save_sensor_data(sensor_dict, field_id)
        
        metadata = output_service.create_response_metadata(field_id, processing_start_time)
        
        logger.info("Legacy upload prediction completed successfully")
        return PredictionResponse(
            healthMapUrl=health_map_url,
            indicesDataUrl=indices_data_url,
            reportUrl=report_url,
            sensorDataUrl=sensor_data_url,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint to verify model loading status and service readiness.
    
    Returns detailed information about:
    - Model availability and loading status
    - Service configuration
    - System readiness
    """
    try:
        model_status = ml_models.get_model_status()
        
        # Determine overall service health
        models_ready = sum(1 for status in model_status.values() if status['loaded'])
        models_available = sum(1 for status in model_status.values() if status['available'])
        
        if models_ready == 0:
            service_status = "unhealthy"
            message = "No models are loaded. Please ensure model files exist in the models directory."
        elif models_ready < models_available:
            service_status = "partial"
            message = f"{models_ready}/{models_available} models loaded successfully."
        else:
            service_status = "healthy"
            message = "All available models loaded successfully."
        
        return {
            "status": service_status,
            "message": message,
            "models": model_status,
            "service_info": {
                "version": "2.0.0",
                "models_ready": models_ready,
                "models_available": models_available,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )

@router.get("/models/status", response_model=List[ModelStatus])
async def get_models_status():
    """
    Get detailed status of all available ML models in the system.
    Useful for monitoring which models are loaded and ready for inference.
    """
    try:
        model_status = ml_models.get_model_status()
        
        models_list = []
        for model_name, status in model_status.items():
            models_list.append(ModelStatus(
                modelName=model_name,
                loaded=status['loaded'],
                modelPath=status['path'],
                type=model_name.upper(),
                description=f"{model_name.title()} model for agricultural analysis",
                lastUpdated=datetime.now().isoformat() if status['loaded'] else None
            ))
        
        return models_list
        
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models status: {str(e)}"
        )

@router.get("/fields", response_model=List[FieldInfo])
async def get_available_fields():
    """
    Get list of available field data for processing.
    This endpoint provides metadata about fields that can be analyzed.
    """
    try:
        # In production, this would connect to your field management database
        sample_fields = [
            FieldInfo(
                fieldId="field_001",
                name="North Wheat Field A",
                location={"lat": 40.7128, "lon": -74.0060},
                area_hectares=25.5,
                crop_type="wheat",
                last_updated="2025-10-04T08:00:00Z",
                data_available=["hyperspectral", "sensor_data", "weather", "soil_analysis"]
            ),
            FieldInfo(
                fieldId="field_002",
                name="South Corn Field B",
                location={"lat": 40.7589, "lon": -73.9851},
                area_hectares=18.3,
                crop_type="corn",
                last_updated="2025-10-04T09:30:00Z",
                data_available=["hyperspectral", "sensor_data", "weather"]
            ),
            FieldInfo(
                fieldId="field_003",
                name="East Soybean Field C",
                location={"lat": 40.7282, "lon": -73.7949},
                area_hectares=32.1,
                crop_type="soybean",
                last_updated="2025-10-04T10:15:00Z",
                data_available=["hyperspectral", "sensor_data", "weather", "irrigation"]
            )
        ]
        
        return sample_fields
        
    except Exception as e:
        logger.error(f"Error getting fields data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving fields data: {str(e)}"
        )

@router.post("/models/reload")
async def reload_models():
    """
    Reload all ML models from their respective file paths.
    Useful for updating models without restarting the service.
    """
    try:
        logger.info("Reloading all ML models...")
        
        # Force reload all models
        ml_models.load_cnn_model()
        ml_models.load_autoencoder_model()
        ml_models.load_lstm_model()
        
        model_status = ml_models.get_model_status()
        models_loaded = sum(1 for status in model_status.values() if status['loaded'])
        
        return {
            "status": "success",
            "message": f"Model reload completed. {models_loaded} models loaded successfully.",
            "models": model_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading models: {str(e)}"
        )