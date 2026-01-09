import tensorflow as tf
import numpy as np
from typing import Any, List, Optional, Dict
import logging
import os
from app.config import settings

logger = logging.getLogger(__name__)

class MultiModalCNNModel:
    """Multi-modal CNN model for agricultural analysis."""
    
    def __init__(self):
        self.cnn_model: Optional[tf.keras.Model] = None
        self.autoencoder_model: Optional[tf.keras.Model] = None
        self.lstm_model: Optional[tf.keras.Model] = None
        self._models_loaded = {
            'cnn': False,
            'autoencoder': False,
            'lstm': False
        }
        
    def ensure_models_loaded(self) -> None:
        """Ensure all available models are loaded."""
        self.load_cnn_model()
        self.load_autoencoder_model()
        self.load_lstm_model()
    
    def load_cnn_model(self) -> None:
        """Load the CNN model for hyperspectral analysis."""
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.cnn_model = tf.keras.models.load_model(settings.MODEL_PATH)
                self._models_loaded['cnn'] = True
                logger.info(f"CNN model loaded successfully from {settings.MODEL_PATH}")
            else:
                logger.warning(f"CNN model file not found at {settings.MODEL_PATH}")
                self.cnn_model = None
                self._models_loaded['cnn'] = False
        except Exception as e:
            logger.error(f"Failed to load CNN model: {str(e)}")
            self.cnn_model = None
            self._models_loaded['cnn'] = False
    
    def load_autoencoder_model(self) -> None:
        """Load the autoencoder model for feature extraction."""
        try:
            if os.path.exists(settings.AUTOENCODER_PATH):
                self.autoencoder_model = tf.keras.models.load_model(settings.AUTOENCODER_PATH)
                self._models_loaded['autoencoder'] = True
                logger.info(f"Autoencoder model loaded successfully from {settings.AUTOENCODER_PATH}")
            else:
                logger.warning(f"Autoencoder model file not found at {settings.AUTOENCODER_PATH}")
                self.autoencoder_model = None
                self._models_loaded['autoencoder'] = False
        except Exception as e:
            logger.error(f"Failed to load autoencoder model: {str(e)}")
            self.autoencoder_model = None
            self._models_loaded['autoencoder'] = False
    
    def load_lstm_model(self) -> None:
        """Load the LSTM model for temporal analysis."""
        try:
            if os.path.exists(settings.LSTM_PATH):
                self.lstm_model = tf.keras.models.load_model(settings.LSTM_PATH)
                self._models_loaded['lstm'] = True
                logger.info(f"LSTM model loaded successfully from {settings.LSTM_PATH}")
            else:
                logger.warning(f"LSTM model file not found at {settings.LSTM_PATH}")
                self.lstm_model = None
                self._models_loaded['lstm'] = False
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {str(e)}")
            self.lstm_model = None
            self._models_loaded['lstm'] = False
    
    def predict_cnn(self, combined_data: np.ndarray) -> List[float]:
        """Make prediction using the CNN model."""
        try:
            if not self._models_loaded['cnn']:
                self.load_cnn_model()
            
            if self.cnn_model is None:
                raise RuntimeError("CNN model not loaded. Please ensure spectral_cnn.h5 exists in the models directory.")
                
            prediction = self.cnn_model.predict(combined_data)
            return prediction.tolist()
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {str(e)}")
            raise RuntimeError(f"CNN prediction failed: {str(e)}")
    
    def extract_features(self, hyperspectral_data: np.ndarray) -> np.ndarray:
        """Extract features using the autoencoder model."""
        try:
            if not self._models_loaded['autoencoder']:
                self.load_autoencoder_model()
            
            if self.autoencoder_model is None:
                logger.warning("Autoencoder model not available, using raw data")
                return hyperspectral_data.flatten()
                
            # Use encoder part of autoencoder
            encoder = tf.keras.Model(
                inputs=self.autoencoder_model.input,
                outputs=self.autoencoder_model.get_layer('encoder_output').output
            )
            features = encoder.predict(hyperspectral_data)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            # Fallback to raw data
            return hyperspectral_data.flatten()
    
    def predict_temporal(self, temporal_data: np.ndarray) -> List[float]:
        """Make temporal prediction using LSTM model."""
        try:
            if not self._models_loaded['lstm']:
                self.load_lstm_model()
            
            if self.lstm_model is None:
                raise RuntimeError("LSTM model not loaded. Please ensure spectral_lstm.h5 exists in the models directory.")
                
            prediction = self.lstm_model.predict(temporal_data)
            return prediction.tolist()
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {str(e)}")
            raise RuntimeError(f"LSTM prediction failed: {str(e)}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        return {
            'cnn': {
                'loaded': self._models_loaded['cnn'],
                'path': settings.MODEL_PATH,
                'available': os.path.exists(settings.MODEL_PATH)
            },
            'autoencoder': {
                'loaded': self._models_loaded['autoencoder'],
                'path': settings.AUTOENCODER_PATH,
                'available': os.path.exists(settings.AUTOENCODER_PATH)
            },
            'lstm': {
                'loaded': self._models_loaded['lstm'],
                'path': settings.LSTM_PATH,
                'available': os.path.exists(settings.LSTM_PATH)
            }
        }
    
    def is_ready(self) -> bool:
        """Check if at least one model is ready for predictions."""
        return any(self._models_loaded.values())

# Global model instance
ml_models = MultiModalCNNModel()