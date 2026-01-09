"""
Data preprocessing utilities for hyperspectral images and sensor data
"""
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging
from io import BytesIO

from ..core.exceptions import ImageProcessingException, DataValidationException
from ..config import settings

logger = logging.getLogger(__name__)


class HyperspectralProcessor:
    """Processor for hyperspectral image data"""
    
    @staticmethod
    def load_tif_image(file_path: str) -> np.ndarray:
        """
        Load and preprocess .tif hyperspectral image
        
        Args:
            file_path: Path to the .tif image file
            
        Returns:
            Preprocessed image array
            
        Raises:
            ImageProcessingException: If image loading or processing fails
        """
        try:
            # Load image using PIL
            image = Image.open(file_path)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Handle different image modes
            if image.mode == 'RGB':
                # Standard RGB image
                pass
            elif image.mode == 'RGBA':
                # Remove alpha channel
                image_array = image_array[:, :, :3]
            elif image.mode == 'L':
                # Grayscale - convert to 3 channels
                image_array = np.stack([image_array] * 3, axis=-1)
            else:
                # Other modes - convert to RGB first
                image = image.convert('RGB')
                image_array = np.array(image)
            
            # Normalize to [0, 1]
            if image_array.dtype == np.uint8:
                image_array = image_array.astype(np.float32) / 255.0
            elif image_array.dtype == np.uint16:
                image_array = image_array.astype(np.float32) / 65535.0
            
            # Resize to standard size (224x224 for most CNN models)
            image_array = HyperspectralProcessor.resize_image(image_array, (224, 224))
            
            logger.info(f"Successfully loaded and preprocessed image: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error loading TIF image {file_path}: {str(e)}")
            raise ImageProcessingException(f"Failed to load TIF image: {str(e)}")
    
    @staticmethod
    def load_npy_data(file_path: str) -> np.ndarray:
        """
        Load and preprocess .npy hyperspectral data
        
        Args:
            file_path: Path to the .npy file
            
        Returns:
            Preprocessed data array
            
        Raises:
            ImageProcessingException: If data loading fails
        """
        try:
            # Load numpy array
            data = np.load(file_path)
            
            # Validate data shape
            if data.ndim < 2:
                raise ValueError(f"Invalid data dimensions: {data.ndim}. Expected at least 2D array.")
            
            # Handle different data shapes
            if data.ndim == 2:
                # 2D data - add channel dimension
                data = np.expand_dims(data, axis=-1)
                data = np.repeat(data, 3, axis=-1)  # Convert to 3 channels
            elif data.ndim == 3:
                # 3D data - check if it's (H, W, C) or (C, H, W)
                if data.shape[0] < data.shape[-1]:
                    # Likely (C, H, W) - transpose to (H, W, C)
                    data = np.transpose(data, (1, 2, 0))
                
                # If too many channels, take first 3
                if data.shape[-1] > 3:
                    data = data[:, :, :3]
                elif data.shape[-1] == 1:
                    # Single channel - repeat to 3 channels
                    data = np.repeat(data, 3, axis=-1)
            
            # Normalize data
            data = HyperspectralProcessor.normalize_data(data)
            
            # Resize to standard size
            data = HyperspectralProcessor.resize_image(data, (224, 224))
            
            logger.info(f"Successfully loaded and preprocessed NPY data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading NPY data {file_path}: {str(e)}")
            raise ImageProcessingException(f"Failed to load NPY data: {str(e)}")
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image array
            target_size: Target size (height, width)
            
        Returns:
            Resized image array
        """
        try:
            if image.shape[:2] != target_size:
                # Use OpenCV for resizing
                resized = cv2.resize(image, (target_size[1], target_size[0]), 
                                   interpolation=cv2.INTER_LINEAR)
                
                # Ensure 3D array
                if resized.ndim == 2:
                    resized = np.expand_dims(resized, axis=-1)
                
                return resized
            return image
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise ImageProcessingException(f"Failed to resize image: {str(e)}")
    
    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data array
        """
        try:
            # Handle different data types
            if data.dtype == np.uint8:
                return data.astype(np.float32) / 255.0
            elif data.dtype == np.uint16:
                return data.astype(np.float32) / 65535.0
            elif data.dtype in [np.float32, np.float64]:
                # Check if already normalized
                if data.min() >= 0 and data.max() <= 1:
                    return data.astype(np.float32)
                else:
                    # Min-max normalization
                    data_min = data.min()
                    data_max = data.max()
                    if data_max - data_min > 0:
                        return ((data - data_min) / (data_max - data_min)).astype(np.float32)
                    else:
                        return np.zeros_like(data, dtype=np.float32)
            else:
                # For other types, use min-max normalization
                data = data.astype(np.float32)
                data_min = data.min()
                data_max = data.max()
                if data_max - data_min > 0:
                    return (data - data_min) / (data_max - data_min)
                else:
                    return np.zeros_like(data)
                    
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise ImageProcessingException(f"Failed to normalize data: {str(e)}")


class SensorDataProcessor:
    """Processor for environmental sensor data"""
    
    @staticmethod
    def load_csv_data(file_path: str) -> np.ndarray:
        """
        Load and preprocess CSV sensor data
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Preprocessed sensor data array
            
        Raises:
            DataValidationException: If data loading or validation fails
        """
        try:
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = [
                'air_temperature', 'humidity', 'soil_moisture', 'wind_speed',
                'rainfall', 'solar_radiation', 'leaf_wetness', 'co2_level', 'ph_level'
            ]
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns in CSV: {missing_columns}")
                # Fill missing columns with default values
                for col in missing_columns:
                    df[col] = SensorDataProcessor.get_default_value(col)
            
            # Select and order columns
            sensor_data = df[required_columns]
            
            # Handle missing values
            sensor_data = sensor_data.fillna(sensor_data.mean())
            
            # Validate data ranges
            sensor_data = SensorDataProcessor.validate_sensor_ranges(sensor_data)
            
            # Normalize sensor data
            normalized_data = SensorDataProcessor.normalize_sensor_data(sensor_data)
            
            logger.info(f"Successfully loaded and preprocessed CSV data: {normalized_data.shape}")
            return normalized_data.values
            
        except Exception as e:
            logger.error(f"Error loading CSV data {file_path}: {str(e)}")
            raise DataValidationException(f"Failed to load CSV data: {str(e)}")
    
    @staticmethod
    def validate_sensor_ranges(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clip sensor data to reasonable ranges
        
        Args:
            data: Input sensor dataframe
            
        Returns:
            Validated dataframe
        """
        # Define reasonable ranges for each sensor
        ranges = {
            'air_temperature': (-10, 50),     # Celsius
            'humidity': (0, 100),             # Percentage
            'soil_moisture': (0, 100),        # Percentage
            'wind_speed': (0, 50),            # m/s
            'rainfall': (0, 200),             # mm
            'solar_radiation': (0, 1500),     # W/m²
            'leaf_wetness': (0, 100),         # Percentage
            'co2_level': (300, 2000),         # ppm
            'ph_level': (0, 14)               # pH scale
        }
        
        for column, (min_val, max_val) in ranges.items():
            if column in data.columns:
                data[column] = data[column].clip(min_val, max_val)
        
        return data
    
    @staticmethod
    def normalize_sensor_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sensor data using min-max scaling
        
        Args:
            data: Input sensor dataframe
            
        Returns:
            Normalized dataframe
        """
        # Define normalization ranges (min, max) for each sensor
        norm_ranges = {
            'air_temperature': (-10, 50),
            'humidity': (0, 100),
            'soil_moisture': (0, 100),
            'wind_speed': (0, 50),
            'rainfall': (0, 200),
            'solar_radiation': (0, 1500),
            'leaf_wetness': (0, 100),
            'co2_level': (300, 2000),
            'ph_level': (0, 14)
        }
        
        normalized_data = data.copy()
        
        for column, (min_val, max_val) in norm_ranges.items():
            if column in normalized_data.columns:
                normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    @staticmethod
    def get_default_value(column: str) -> float:
        """
        Get default value for missing sensor data
        
        Args:
            column: Column name
            
        Returns:
            Default value
        """
        defaults = {
            'air_temperature': 20.0,    # 20°C
            'humidity': 60.0,           # 60%
            'soil_moisture': 50.0,      # 50%
            'wind_speed': 2.0,          # 2 m/s
            'rainfall': 0.0,            # 0 mm
            'solar_radiation': 500.0,   # 500 W/m²
            'leaf_wetness': 30.0,       # 30%
            'co2_level': 400.0,         # 400 ppm
            'ph_level': 7.0             # pH 7 (neutral)
        }
        
        return defaults.get(column, 0.0)


class DataCombiner:
    """Utility class for combining different data types"""
    
    @staticmethod
    def combine_image_and_sensor_data(image_data: np.ndarray, sensor_data: np.ndarray) -> np.ndarray:
        """
        Combine image and sensor data for model input
        
        Args:
            image_data: Preprocessed image data
            sensor_data: Preprocessed sensor data
            
        Returns:
            Combined data array
        """
        try:
            # Flatten image data
            image_flattened = image_data.flatten()
            
            # Flatten sensor data if needed
            if sensor_data.ndim > 1:
                sensor_flattened = sensor_data.flatten()
            else:
                sensor_flattened = sensor_data
            
            # Concatenate data
            combined_data = np.concatenate([image_flattened, sensor_flattened])
            
            logger.info(f"Combined data shape: {combined_data.shape}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            raise DataValidationException(f"Failed to combine data: {str(e)}")
    
    @staticmethod
    def prepare_model_input(image_data: np.ndarray, sensor_data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Prepare final input for model prediction
        
        Args:
            image_data: Preprocessed image data
            sensor_data: Optional preprocessed sensor data
            
        Returns:
            Dictionary with model inputs
        """
        try:
            # Ensure image data has batch dimension
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)
            
            model_input = {'image': image_data}
            
            if sensor_data is not None:
                # Ensure sensor data has batch dimension
                if sensor_data.ndim == 1:
                    sensor_data = np.expand_dims(sensor_data, axis=0)
                model_input['sensor'] = sensor_data
            
            logger.info(f"Prepared model input with keys: {list(model_input.keys())}")
            return model_input
            
        except Exception as e:
            logger.error(f"Error preparing model input: {str(e)}")
            raise DataValidationException(f"Failed to prepare model input: {str(e)}")