import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Dict

class MultimodalFusionNetwork:
    """Neural network for fusing spectral, sensor, and temporal data"""
    
    def __init__(self, spectral_shape: tuple, sensor_features: int, 
                 temporal_features: int, config: Dict):
        self.spectral_shape = spectral_shape
        self.sensor_features = sensor_features
        self.temporal_features = temporal_features
        self.config = config
        self.model = None
        
    def build_spectral_branch(self, inputs):
        """Build spectral data processing branch"""
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        return x
    
    def build_sensor_branch(self, inputs):
        """Build sensor data processing branch"""
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        return x
    
    def build_temporal_branch(self, inputs):
        """Build temporal data processing branch"""
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(32)(x)
        x = layers.Dense(64, activation='relu')(x)
        return x
    
    def build_model(self):
        """Build multimodal fusion network"""
        # Input branches
        spectral_input = keras.Input(shape=self.spectral_shape, name='spectral_input')
        sensor_input = keras.Input(shape=(self.sensor_features,), name='sensor_input')
        temporal_input = keras.Input(shape=(None, self.temporal_features), name='temporal_input')
        
        # Process each modality
        spectral_features = self.build_spectral_branch(spectral_input)
        sensor_features = self.build_sensor_branch(sensor_input)
        temporal_features = self.build_temporal_branch(temporal_input)
        
        # Attention mechanism for feature fusion
        spectral_attention = layers.Dense(1, activation='sigmoid')(spectral_features)
        sensor_attention = layers.Dense(1, activation='sigmoid')(sensor_features)
        temporal_attention = layers.Dense(1, activation='sigmoid')(temporal_features)
        
        # Weighted feature fusion
        weighted_spectral = layers.multiply([spectral_features, spectral_attention])
        weighted_sensor = layers.multiply([sensor_features, sensor_attention])
        weighted_temporal = layers.multiply([temporal_features, temporal_attention])
        
        # Concatenate all features
        fused_features = layers.concatenate([weighted_spectral, weighted_sensor, weighted_temporal])
        
        # Final prediction layers
        x = layers.Dense(256, activation='relu')(fused_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-output predictions
        health_status = layers.Dense(4, activation='softmax', name='health_status')(x)  # Healthy, Stress, Disease, Pest
        severity_score = layers.Dense(1, activation='sigmoid', name='severity_score')(x)
        risk_prediction = layers.Dense(1, activation='sigmoid', name='risk_prediction')(x)
        
        self.model = keras.Model(
            inputs=[spectral_input, sensor_input, temporal_input],
            outputs=[health_status, severity_score, risk_prediction],
            name='MultimodalFusionNetwork'
        )
        
        return self.model
    
    def compile_model(self):
        """Compile multimodal model"""
        self.model.compile(
            optimizer='adam',
            loss={
                'health_status': 'categorical_crossentropy',
                'severity_score': 'mse',
                'risk_prediction': 'binary_crossentropy'
            },
            loss_weights={
                'health_status': 1.0,
                'severity_score': 0.5,
                'risk_prediction': 0.3
            },
            metrics={
                'health_status': ['accuracy'],
                'severity_score': ['mae'],
                'risk_prediction': ['accuracy']
            }
        )
    
    def predict_comprehensive(self, spectral_data: np.ndarray, 
                            sensor_data: np.ndarray, 
                            temporal_data: np.ndarray) -> Dict:
        """Make comprehensive predictions using all modalities"""
        predictions = self.model.predict([spectral_data, sensor_data, temporal_data])
        
        health_probs = predictions[0][0]
        severity = predictions[1][0][0]
        risk = predictions[2][0][0]
        
        health_classes = ['healthy', 'stressed', 'diseased', 'pest_damage']
        predicted_class = health_classes[np.argmax(health_probs)]
        
        return {
            'health_status': {
                'predicted_class': predicted_class,
                'confidence': float(np.max(health_probs)),
                'probabilities': {cls: float(prob) for cls, prob in zip(health_classes, health_probs)}
            },
            'severity_score': float(severity),
            'risk_score': float(risk),
            'alert_level': self._determine_alert_level(severity, risk)
        }
    
    def _determine_alert_level(self, severity: float, risk: float) -> str:
        """Determine alert level based on severity and risk scores"""
        combined_score = (severity + risk) / 2
        
        if combined_score > 0.8:
            return 'CRITICAL'
        elif combined_score > 0.6:
            return 'HIGH'
        elif combined_score > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

class CrossModalAttention:
    """Cross-modal attention mechanism for better feature fusion"""
    
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        
    def build_attention_layer(self, query_input, key_input, value_input):
        """Build cross-modal attention layer"""
        # Linear transformations
        query = layers.Dense(self.d_model)(query_input)
        key = layers.Dense(self.d_model)(key_input)
        value = layers.Dense(self.d_model)(value_input)
        
        # Attention weights
        attention_weights = layers.Dense(1, activation='softmax')(
            layers.concatenate([query, key])
        )
        
        # Apply attention
        attended_value = layers.multiply([value, attention_weights])
        
        return attended_value, attention_weights

class EnsemblePredictor:
    """Ensemble multiple models for robust predictions"""
    
    def __init__(self, models: List):
        self.models = models
        self.weights = None
        
    def fit_ensemble_weights(self, X_val, y_val):
        """Learn ensemble weights based on validation performance"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X_val)
            predictions.append(pred)
        
        # Simple averaging for now (could use more sophisticated methods)
        self.weights = np.ones(len(self.models)) / len(self.models)
        
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Calculate prediction uncertainty
        uncertainty = np.std(predictions, axis=0)
        
        return ensemble_pred, uncertainty

class FeatureFusionModule:
    """Advanced feature fusion with learnable weights"""
    
    def __init__(self, feature_dims: List[int]):
        self.feature_dims = feature_dims
        
    def build_fusion_layer(self, feature_inputs: List):
        """Build adaptive feature fusion layer"""
        # Normalize features to same dimension
        normalized_features = []
        for i, features in enumerate(feature_inputs):
            normalized = layers.Dense(256, activation='relu')(features)
            normalized_features.append(normalized)
        
        # Learn fusion weights
        fusion_weights = []
        for features in normalized_features:
            weight = layers.Dense(1, activation='sigmoid')(features)
            fusion_weights.append(weight)
        
        # Normalize weights
        weight_sum = layers.add(fusion_weights)
        normalized_weights = [layers.divide([w, weight_sum]) for w in fusion_weights]
        
        # Apply weights and fuse
        weighted_features = [layers.multiply([feat, weight]) 
                           for feat, weight in zip(normalized_features, normalized_weights)]
        
        fused = layers.add(weighted_features)
        
        return fused, normalized_weights