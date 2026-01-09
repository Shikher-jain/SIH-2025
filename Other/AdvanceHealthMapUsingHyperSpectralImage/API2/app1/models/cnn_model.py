"""
CNN Model wrapper for spectral health mapping
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path
import pickle

from ..core.exceptions import ModelNotLoadedException, ModelPredictionException
from ..config import settings

logger = logging.getLogger(__name__)


class SpectralCNNModel:
    """CNN Model for hyperspectral image analysis and crop health prediction"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the CNN model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[tf.keras.Model] = None
        self.is_loaded = False
        self.input_shape = (224, 224, 3)
        self.num_classes = 5  # Health categories: Excellent, Good, Fair, Poor, Critical
        
        # Load model if path exists
        if Path(self.model_path).exists():
            self.load_model()
        else:
            logger.warning(f"Model file not found at {self.model_path}. Model needs to be trained or loaded.")
    
    def create_model(self) -> tf.keras.Model:
        """
        Create a new CNN model architecture
        
        Returns:
            Compiled Keras model
        """
        try:
            model = Sequential([
                # First Convolutional Block
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                
                # Second Convolutional Block
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                
                # Third Convolutional Block
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                
                # Fourth Convolutional Block
                Conv2D(256, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                
                # Global Average Pooling instead of Flatten to reduce parameters
                tf.keras.layers.GlobalAveragePooling2D(),
                
                # Dense layers
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                
                # Output layers for different predictions
                Dense(64, activation='relu', name='features'),
                
                # Multiple outputs
                Dense(self.num_classes, activation='softmax', name='health_class'),
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Created new CNN model architecture")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise ModelPredictionException(f"Failed to create model: {str(e)}")
    
    def load_model(self) -> bool:
        """
        Load pre-trained model from file
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not Path(self.model_path).exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load the model
            self.model = load_model(self.model_path)
            self.is_loaded = True
            
            logger.info(f"Successfully loaded model from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def save_model(self, save_path: Optional[str] = None) -> bool:
        """
        Save the current model to file
        
        Args:
            save_path: Path to save the model
            
        Returns:
            True if model saved successfully, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            save_path = save_path or self.model_path
            
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            self.model.save(save_path)
            
            logger.info(f"Successfully saved model to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def predict(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on hyperspectral image data
        
        Args:
            image_data: Preprocessed image data
            
        Returns:
            Dictionary containing predictions and confidence scores
            
        Raises:
            ModelNotLoadedException: If model is not loaded
            ModelPredictionException: If prediction fails
        """
        if not self.is_loaded or self.model is None:
            # Try to load model first
            if not self.load_model():
                raise ModelNotLoadedException("Model is not loaded and could not be loaded from file")
        
        try:
            # Ensure input has correct shape
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)
            
            # Validate input shape
            expected_shape = (None,) + self.input_shape
            if image_data.shape[1:] != self.input_shape:
                raise ValueError(f"Invalid input shape: {image_data.shape}. Expected: {expected_shape}")
            
            # Make prediction
            predictions = self.model.predict(image_data, verbose=0)
            
            # If model has multiple outputs, handle accordingly
            if isinstance(predictions, list):
                health_probs = predictions[0]  # Assuming first output is health classification
            else:
                health_probs = predictions
            
            # Get health class prediction
            health_class_idx = np.argmax(health_probs[0])
            health_confidence = float(health_probs[0][health_class_idx])
            
            # Map class index to health category
            health_categories = ['Critical', 'Poor', 'Fair', 'Good', 'Excellent']
            health_category = health_categories[health_class_idx]
            
            # Calculate overall health score (0-100)
            health_score = float(health_class_idx * 25 + health_confidence * 25)
            
            # Extract features for analysis
            feature_extractor = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('features').output
            )
            features = feature_extractor.predict(image_data, verbose=0)
            
            # Generate vegetation indices (simulated based on features)
            indices = self._calculate_vegetation_indices(features[0])
            
            # Generate stress indicators and recommendations
            stress_indicators = self._detect_stress_indicators(health_score, indices)
            recommendations = self._generate_recommendations(health_category, stress_indicators)
            
            # Calculate risk assessments
            disease_risk = max(0, 100 - health_score - np.random.normal(0, 10))
            pest_risk = max(0, 80 - health_score - np.random.normal(0, 15))
            
            result = {
                'health_score': health_score,
                'health_category': health_category,
                'health_confidence': health_confidence,
                'vegetation_indices': indices,
                'stress_indicators': stress_indicators,
                'recommendations': recommendations,
                'disease_risk': max(0, min(100, disease_risk)),
                'pest_risk': max(0, min(100, pest_risk)),
                'model_confidence': health_confidence
            }
            
            logger.info(f"Prediction completed. Health category: {health_category}, Score: {health_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ModelPredictionException(f"Prediction failed: {str(e)}")
    
    def _calculate_vegetation_indices(self, features: np.ndarray) -> Dict[str, float]:
        """
        Calculate vegetation indices based on extracted features
        
        Args:
            features: Extracted features from the model
            
        Returns:
            Dictionary of vegetation indices
        """
        # Simulate vegetation indices calculation based on features
        # In a real implementation, these would be calculated from spectral bands
        
        # Normalize features
        features_norm = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # Simulate NDVI (Normalized Difference Vegetation Index)
        ndvi = float(np.mean(features_norm[:32]) * 0.8 + 0.1)  # Range: 0.1-0.9
        
        # Simulate other indices based on features
        evi = float(ndvi * 0.85 + np.random.normal(0, 0.05))  # Enhanced Vegetation Index
        savi = float(ndvi * 0.9 + np.random.normal(0, 0.03))   # Soil Adjusted Vegetation Index
        ndwi = float(np.mean(features_norm[32:48]) * 0.6 + 0.2)  # Water Index
        pri = float(np.mean(features_norm[48:]) * 0.4 - 0.2)     # Photochemical Reflectance Index
        
        # Derived indices
        chlorophyll = float(ndvi * 45 + 5)  # mg/g
        lai = float(ndvi * 6 + 0.5)         # Leaf Area Index
        water_stress = float((1 - ndwi) * 100)  # Water stress percentage
        
        return {
            'ndvi': max(0, min(1, ndvi)),
            'evi': max(0, min(1, evi)),
            'savi': max(0, min(1, savi)),
            'ndwi': max(0, min(1, ndwi)),
            'pri': max(-0.2, min(0.2, pri)),
            'chlorophyll_content': max(0, chlorophyll),
            'leaf_area_index': max(0, lai),
            'water_stress_index': max(0, min(100, water_stress))
        }
    
    def _detect_stress_indicators(self, health_score: float, indices: Dict[str, float]) -> list:
        """
        Detect stress indicators based on health score and vegetation indices
        
        Args:
            health_score: Overall health score
            indices: Vegetation indices
            
        Returns:
            List of stress indicators
        """
        stress_indicators = []
        
        if health_score < 30:
            stress_indicators.append("Severe crop stress detected")
        elif health_score < 50:
            stress_indicators.append("Moderate crop stress")
        elif health_score < 70:
            stress_indicators.append("Mild stress indicators")
        
        if indices['ndvi'] < 0.3:
            stress_indicators.append("Low vegetation vigor")
        
        if indices['water_stress_index'] > 70:
            stress_indicators.append("Water stress detected")
        
        if indices['chlorophyll_content'] < 20:
            stress_indicators.append("Chlorophyll deficiency")
        
        if indices['leaf_area_index'] < 2:
            stress_indicators.append("Low leaf area coverage")
        
        return stress_indicators
    
    def _generate_recommendations(self, health_category: str, stress_indicators: list) -> list:
        """
        Generate management recommendations based on health status
        
        Args:
            health_category: Health category
            stress_indicators: List of stress indicators
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if health_category == "Critical":
            recommendations.extend([
                "Immediate intervention required",
                "Consult agricultural specialist",
                "Consider soil and water testing",
                "Implement intensive monitoring"
            ])
        elif health_category == "Poor":
            recommendations.extend([
                "Increase monitoring frequency",
                "Check irrigation system",
                "Consider nutrient supplementation",
                "Inspect for pest and disease signs"
            ])
        elif health_category == "Fair":
            recommendations.extend([
                "Monitor crop development closely",
                "Optimize irrigation schedule",
                "Regular pest scouting",
                "Consider foliar nutrition"
            ])
        elif health_category == "Good":
            recommendations.extend([
                "Maintain current management practices",
                "Continue regular monitoring",
                "Prepare for harvest optimization"
            ])
        else:  # Excellent
            recommendations.extend([
                "Excellent crop health - maintain practices",
                "Monitor for optimal harvest timing",
                "Document successful practices"
            ])
        
        # Add specific recommendations based on stress indicators
        if "Water stress detected" in stress_indicators:
            recommendations.append("Increase irrigation frequency")
        
        if "Chlorophyll deficiency" in stress_indicators:
            recommendations.append("Apply nitrogen fertilizer")
        
        if "Low vegetation vigor" in stress_indicators:
            recommendations.append("Soil nutrient analysis recommended")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_summary': str(self.model.summary()) if self.model else None
        }