import numpy as np
import pandas as pd
from PIL import Image
import io
import requests
import tempfile
import os
from fastapi import UploadFile, HTTPException
from typing import Tuple, Dict, Any
import logging
from app.schemas.prediction import SensorData
from app.config import settings

logger = logging.getLogger(__name__)

class AdvancedDataPreprocessor:
    """Advanced data preprocessing for agricultural ML models."""
    
    def __init__(self):
        self.normalization_stats = {}
    
    def download_hyperspectral_image(self, url: str) -> np.ndarray:
        """Download and process hyperspectral image from URL."""
        try:
            logger.info(f"Downloading hyperspectral image from {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Handle different file types
            if url.lower().endswith('.npy'):
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(response.content)
                    tmp.flush()
                    data = np.load(tmp.name)
            elif url.lower().endswith(('.tif', '.tiff')):
                image = Image.open(io.BytesIO(response.content))
                data = np.array(image)
            elif url.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(io.BytesIO(response.content))
                data = np.array(image)
            else:
                # Try as generic image
                image = Image.open(io.BytesIO(response.content))
                data = np.array(image)
            
            # Normalize and validate data
            data = self._validate_and_normalize_image(data)
            return data
                
        except requests.RequestException as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download hyperspectral image: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error processing downloaded image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing hyperspectral image: {str(e)}"
            )
    
    def _validate_and_normalize_image(self, data: np.ndarray) -> np.ndarray:
        """Validate and normalize image data."""
        # Ensure 3D array (height, width, channels)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        
        # Resize if necessary to match model input shape
        if data.shape[:2] != settings.IMAGE_INPUT_SHAPE[:2]:
            from PIL import Image as PILImage
            if data.shape[-1] == 1:
                img = PILImage.fromarray(data.squeeze(), mode='L')
            else:
                img = PILImage.fromarray(data.astype(np.uint8))
            img = img.resize(settings.IMAGE_INPUT_SHAPE[:2])
            data = np.array(img)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=-1)
        
        # Normalize to [0, 1]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if data.max() > 1.0:
            data = data / 255.0
        
        return data
    
    def process_sensor_data(self, sensor_data: SensorData) -> np.ndarray:
        """Process and normalize sensor data."""
        try:
            # Convert to dictionary and extract numeric values
            data_dict = sensor_data.dict()
            timestamp = data_dict.pop('timestamp', None)
            
            # Define expected sensor order and ranges for normalization
            sensor_specs = {
                'airTemperature': {'min': -20, 'max': 50},      # Celsius
                'humidity': {'min': 0, 'max': 100},             # Percentage
                'soilMoisture': {'min': 0, 'max': 100},         # Percentage
                'windSpeed': {'min': 0, 'max': 30},             # m/s
                'rainfall': {'min': 0, 'max': 100},             # mm
                'solarRadiation': {'min': 0, 'max': 1200},      # W/m²
                'leafWetness': {'min': 0, 'max': 100},          # Percentage
                'co2Level': {'min': 300, 'max': 1000},          # ppm
                'phLevel': {'min': 4, 'max': 10}                # pH scale
            }
            
            # Normalize each sensor value
            normalized_values = []
            for key, value in data_dict.items():
                if key in sensor_specs:
                    spec = sensor_specs[key]
                    # Min-max normalization
                    normalized_value = (value - spec['min']) / (spec['max'] - spec['min'])
                    # Clip to [0, 1] range
                    normalized_value = np.clip(normalized_value, 0, 1)
                    normalized_values.append(normalized_value)
                else:
                    # If unknown sensor, use as-is but normalize
                    normalized_values.append(value / 100.0 if value > 10 else value)
            
            return np.array(normalized_values, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing sensor data: {str(e)}"
            )
    
    def combine_multimodal_data(self, hyperspectral_data: np.ndarray, 
                               sensor_data: np.ndarray) -> np.ndarray:
        """Combine hyperspectral and sensor data for model input."""
        try:
            # Flatten hyperspectral data
            hyperspectral_flat = hyperspectral_data.flatten()
            
            # Ensure sensor data is 1D
            sensor_flat = sensor_data.flatten()
            
            # Combine features
            combined = np.concatenate([hyperspectral_flat, sensor_flat])
            
            # Reshape for model input (batch_size=1)
            return combined.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error combining multimodal data: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error combining data: {str(e)}"
            )
    
    def prepare_temporal_data(self, sensor_readings: list, 
                             sequence_length: int = 24) -> np.ndarray:
        """Prepare temporal sensor data for LSTM models."""
        try:
            # If we have multiple readings, use them for temporal analysis
            if len(sensor_readings) >= sequence_length:
                temporal_data = np.array(sensor_readings[-sequence_length:])
            else:
                # Pad with the last reading if insufficient data
                temporal_data = np.array(sensor_readings)
                padding_length = sequence_length - len(sensor_readings)
                if padding_length > 0:
                    last_reading = sensor_readings[-1] if sensor_readings else np.zeros(settings.SENSOR_FEATURES)
                    padding = np.tile(last_reading, (padding_length, 1))
                    temporal_data = np.concatenate([padding, temporal_data])
            
            # Reshape for LSTM input (batch_size, sequence_length, features)
            return temporal_data.reshape(1, sequence_length, -1)
            
        except Exception as e:
            logger.error(f"Error preparing temporal data: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error preparing temporal data: {str(e)}"
            )
    
    # Legacy methods for backward compatibility
    def load_tif_image(self, file: UploadFile) -> np.ndarray:
        """Load and preprocess TIF image file."""
        try:
            image_data = io.BytesIO(file.file.read())
            image = Image.open(image_data)
            data = np.array(image)
            return self._validate_and_normalize_image(data)
        except Exception as e:
            logger.error(f"Error processing TIF file: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing TIF file: {str(e)}"
            )
    
    def load_npy_file(self, file: UploadFile) -> np.ndarray:
        """Load NPY file."""
        try:
            file_data = np.load(file.file)
            return self._validate_and_normalize_image(file_data)
        except Exception as e:
            logger.error(f"Error processing NPY file: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing NPY file: {str(e)}"
            )
    
    def load_csv_file(self, file: UploadFile) -> np.ndarray:
        """Load and convert CSV file to NumPy array."""
        try:
            df = pd.read_csv(file.file)
            # Basic preprocessing for CSV data
            df = df.fillna(df.mean())  # Fill missing values
            return df.values.astype(np.float32)
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing CSV file: {str(e)}"
            )
    
    def combine_legacy_data(self, image_data: np.ndarray, npy_data: np.ndarray, 
                           sensor_data: np.ndarray) -> np.ndarray:
        """Legacy method for combining different data types."""
        try:
            combined = np.concatenate([
                image_data.flatten(),
                npy_data.flatten(),
                sensor_data.flatten()
            ])
            return combined.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error combining legacy data: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error combining data: {str(e)}"
            )

# Global preprocessor instance
preprocessor = AdvancedDataPreprocessor()