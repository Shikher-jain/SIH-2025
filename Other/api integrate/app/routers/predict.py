from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.cnn_model import cnn_model
from app.utils.preprocessing import preprocessor
from app.schemas.prediction import PredictionResponse, ErrorResponse, PredictionRequest, SensorData
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

@router.post("/files", response_model=PredictionResponse)
async def predict_multimodal_files(
    tif_file: UploadFile = File(..., description="TIF image file"),
    npy_file: UploadFile = File(..., description="NumPy array file (.npy)"),
    csv_file: UploadFile = File(..., description="CSV sensor data file"),
):
    """
    Multi-modal prediction endpoint that accepts TIF image, NPY array, and CSV data files.
    
    - **tif_file**: Upload a TIF image file
    - **npy_file**: Upload a NumPy array file (.npy format)
    - **csv_file**: Upload a CSV file containing sensor data
    
    Returns prediction results from the CNN model.
    """
    start_time = time.time()
    
    try:
        # Validate file types
        if not tif_file.filename.lower().endswith(('.tif', '.tiff')):
            raise HTTPException(status_code=400, detail="TIF file must have .tif or .tiff extension")
        
        if not npy_file.filename.lower().endswith('.npy'):
            raise HTTPException(status_code=400, detail="NumPy file must have .npy extension")
            
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Sensor data file must have .csv extension")
        
        # Process each file
        logger.info("Processing uploaded files...")
        
        # Load and preprocess data
        image_data = preprocessor.load_tif_image(tif_file)
        npy_data = preprocessor.load_npy_file(npy_file)
        sensor_data = preprocessor.load_csv_file(csv_file)
        
        # Combine data for model input
        combined_data = preprocessor.combine_data(image_data, npy_data, sensor_data)
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = cnn_model.predict(combined_data)
        
        processing_time = time.time() - start_time
        logger.info("Prediction completed successfully")
        
        return PredictionResponse(
            prediction=prediction,
            processingTime=processing_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/", response_model=PredictionResponse)
async def predict_with_urls(request: PredictionRequest):
    """
    JSON-based prediction endpoint for JavaScript backend integration.
    
    Accepts:
    - Field ID and User ID for tracking
    - Hyperspectral image URL (will be downloaded)
    - Sensor data as JSON object
    
    Returns prediction results from the CNN model.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting prediction for field {request.fieldId}")
        
        # Download and process hyperspectral image
        hyperspectral_data = preprocessor.download_hyperspectral_image(str(request.hyperSpectralImageUrl))
        
        # Process sensor data
        sensor_array = preprocessor.process_sensor_data(request.sensorData)
        
        # For now, create dummy npy data since we only have hyperspectral and sensor
        # In production, you might want to modify this based on your actual data structure
        dummy_npy = hyperspectral_data  # Use the same data or provide separate NPY URL
        
        # Combine data for model input
        combined_data = preprocessor.combine_data(hyperspectral_data, dummy_npy, sensor_array)
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = cnn_model.predict(combined_data)
        
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed successfully for field {request.fieldId}")
        
        return PredictionResponse(
            prediction=prediction,
            fieldId=request.fieldId,
            processingTime=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint to verify the model is loaded and ready."""
    try:
        model_ready = cnn_model.is_model_ready()
        
        if not model_ready:
            return {
                "status": "warning",
                "model_loaded": False,
                "message": "Model not loaded. Please ensure model.h5 exists in the project directory."
            }
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "message": "Prediction service is ready"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )

@router.get("/status")
async def get_api_status():
    """Get API status and information for JavaScript integration."""
    try:
        model_ready = cnn_model.is_model_ready()
        
        return {
            "status": "online",
            "model_loaded": model_ready,
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "predict_json": "/predict/",
                "predict_files": "/predict/files",
                "health_check": "/predict/health",
                "status": "/predict/status"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service status check failed: {str(e)}"
        )