import numpy as np
import pandas as pd
from PIL import Image
import io
import requests
from fastapi import UploadFile, HTTPException
from typing import Tuple, Dict, Any, Optional, TYPE_CHECKING
import logging
import tempfile
import os
from datetime import datetime

if TYPE_CHECKING:
    from app.schemas.prediction import SensorData

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles preprocessing of different data types for the CNN model."""
    
    @staticmethod
    def download_hyperspectral_image(url: str) -> np.ndarray:
        """
        Download hyperspectral image from URL.
        
        Args:
            url: URL to download the image from
            
        Returns:
            NumPy array of the hyperspectral image
        """
        try:
            logger.info(f"Downloading hyperspectral image from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Handle different file types
            if url.lower().endswith('.npy'):
                # Load NPY file from bytes
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(response.content)
                    tmp.flush()
                    return np.load(tmp.name)
            elif url.lower().endswith(('.tif', '.tiff')):
                # Load TIF image
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
            else:
                # Try as generic image
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
                
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
    
    @staticmethod
    def process_sensor_data(sensor_data: 'SensorData') -> np.ndarray:
        """
        Convert sensor data to NumPy array.
        
        Args:
            sensor_data: Pydantic model containing sensor readings
            
        Returns:
            NumPy array of sensor data
        """
        try:
            # Convert to dictionary and extract numeric values
            data_dict = sensor_data.dict()
            # Remove timestamp as it's not numeric
            timestamp = data_dict.pop('timestamp', None)
            
            # Convert to array
            values = list(data_dict.values())
            return np.array(values, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing sensor data: {str(e)}"
            )
    
    @staticmethod
    def load_tif_image(file: UploadFile) -> np.ndarray:
        """
        Load and preprocess TIF image file.
        
        Args:
            file: Uploaded TIF file
            
        Returns:
            NumPy array of the image
        """
        try:
            image_data = io.BytesIO(file.file.read())
            image = Image.open(image_data)
            return np.array(image)
        except Exception as e:
            logger.error(f"Error processing TIF file: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing TIF file: {str(e)}"
            )
    
    @staticmethod
    def load_npy_file(file: UploadFile) -> np.ndarray:
        """
        Load NPY file.
        
        Args:
            file: Uploaded NPY file
            
        Returns:
            NumPy array from the file
        """
        try:
            file_data = np.load(file.file)
            return file_data
        except Exception as e:
            logger.error(f"Error processing NPY file: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing NPY file: {str(e)}"
            )
    
    @staticmethod
    def load_csv_file(file: UploadFile) -> np.ndarray:
        """
        Load and convert CSV file to NumPy array.
        
        Args:
            file: Uploaded CSV file
            
        Returns:
            NumPy array of CSV data
        """
        try:
            df = pd.read_csv(file.file)
            return df.values
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing CSV file: {str(e)}"
            )
    
    @staticmethod
    def combine_data(image_data: np.ndarray, npy_data: np.ndarray, sensor_data: np.ndarray) -> np.ndarray:
        """
        Combine image, npy, and sensor data into model input format.
        
        Args:
            image_data: Processed image data
            npy_data: NumPy array data  
            sensor_data: Sensor data from CSV
            
        Returns:
            Combined and reshaped data ready for model input
        """
        try:
            # Flatten all inputs and concatenate
            combined = np.concatenate([
                image_data.flatten(),
                npy_data.flatten(),
                sensor_data.flatten()
            ])
            
            # Reshape for model input (batch_size=1)
            return combined.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error combining data: {str(e)}"
            )
    
    @staticmethod
    def combine_legacy_data(image_data: np.ndarray, npy_data: np.ndarray, 
                           sensor_data: np.ndarray) -> np.ndarray:
        """
        Legacy method: Combine different data types into model input format.
        
        Args:
            image_data: Processed image data
            npy_data: NumPy array data
            sensor_data: Sensor data from CSV
            
        Returns:
            Combined and reshaped data ready for model input
        """
        try:
            # Flatten all inputs and concatenate
            combined = np.concatenate([
                image_data.flatten(),
                npy_data.flatten(),
                sensor_data.flatten()
            ])
            
            # Reshape for model input (batch_size=1)
            return combined.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error combining data: {str(e)}"
            )

# Global preprocessor instance
preprocessor = DataPreprocessor()