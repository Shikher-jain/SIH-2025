import tensorflow as tf
import numpy as np
from typing import Any, List, Optional
import logging
from app.config import settings
import os

logger = logging.getLogger(__name__)

class CNNMultiModalModel:
    """CNN model for multi-modal prediction using TIF, NPY, and CSV data."""
    
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self._model_loaded = False
        
    def ensure_model_loaded(self) -> None:
        """Ensure the model is loaded before making predictions."""
        if not self._model_loaded:
            self.load_model()
    
    def load_model(self) -> None:
        """Load the pre-trained model."""
        try:
            if not os.path.exists(settings.MODEL_PATH):
                logger.warning(f"Model file not found at {settings.MODEL_PATH}. Model will be None until loaded.")
                self.model = None
                self._model_loaded = False
                return
                
            self.model = tf.keras.models.load_model(settings.MODEL_PATH)
            self._model_loaded = True
            logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.warning("Model will be None until properly loaded.")
            self.model = None
            self._model_loaded = False
    
    def predict(self, combined_data: np.ndarray) -> List[float]:
        """Make prediction using the loaded model."""
        try:
            self.ensure_model_loaded()
            
            if self.model is None:
                raise RuntimeError("Model not loaded. Please ensure model.h5 exists in the project directory.")
                
            prediction = self.model.predict(combined_data)
            return prediction.tolist()
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def is_model_ready(self) -> bool:
        """Check if the model is ready for predictions."""
        return self._model_loaded and self.model is not None

# Global model instance (with lazy loading)
cnn_model = CNNMultiModalModel()