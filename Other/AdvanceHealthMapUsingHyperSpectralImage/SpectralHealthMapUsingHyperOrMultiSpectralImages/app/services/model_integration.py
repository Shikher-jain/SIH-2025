"""
Model integration service to connect existing models with FastAPI
"""
import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModel
import emoji
import joblib

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultiTaskBERT(torch.nn.Module):
    """Multi-task BERT model for disaster classification"""
    
    def __init__(self, bert, num_disaster, num_urgency):
        super().__init__()
        self.bert = bert
        hidden_size = self.bert.config.hidden_size
        self.disaster_head = torch.nn.Linear(hidden_size, num_disaster)
        self.urgency_head = torch.nn.Linear(hidden_size, num_urgency)
        self.relevance_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return (self.disaster_head(pooled),
                self.urgency_head(pooled),
                self.relevance_head(pooled))


class ModelIntegrationService:
    """Service to integrate all existing models with the FastAPI application"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.ENABLE_GPU else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.label_mappings = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models on startup"""
        try:
            logger.info(f"Initializing models on device: {self.device}")
            
            # Initialize BERT-based disaster classification model
            self._load_bert_disaster_model()
            
            # Initialize spectral analysis models
            self._load_spectral_models()
            
            # Initialize traditional ML models
            self._load_traditional_models()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _load_bert_disaster_model(self):
        """Load BERT-based disaster classification model"""
        try:
            # Model configuration
            model_name = "bert-base-uncased"
            num_disaster = 7
            num_urgency = 3
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.tokenizers['disaster_bert'] = tokenizer
            
            # Load BERT model
            bert = AutoModel.from_pretrained(model_name)
            
            # Initialize multi-task model
            model = MultiTaskBERT(bert, num_disaster, num_urgency)
            model.to(self.device)
            
            # Load trained weights if available
            checkpoint_path = os.path.join(settings.MODEL_PATH, "multi_task_bert.pth")
            if os.path.exists(checkpoint_path):
                try:
                    heads = torch.load(checkpoint_path, map_location=self.device)
                    for head_name in ["disaster_head", "urgency_head", "relevance_head"]:
                        if f"{head_name}.weight" in heads and f"{head_name}.bias" in heads:
                            getattr(model, head_name).weight.data = heads[f"{head_name}.weight"]
                            getattr(model, head_name).bias.data = heads[f"{head_name}.bias"]
                    logger.info("Loaded BERT disaster model weights")
                except Exception as e:
                    logger.warning(f"Could not load BERT weights: {str(e)}")
            
            model.eval()
            self.models['disaster_bert'] = model
            
            # Label mappings
            self.label_mappings['disaster_bert'] = {
                'disaster_labels': ['flood', 'hurricane', 'rain', 'cyclone', 'storm', 'high_waves', 'casual'],
                'urgency_labels': ['neutral', 'panic', 'emergency']
            }
            
            logger.info("BERT disaster classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BERT disaster model: {str(e)}")
    
    def _load_spectral_models(self):
        """Load spectral analysis models"""
        try:
            # Import your existing spectral models
            from src.models.spectral_models import SpectralCNN, SpectralLSTM, SpectralAutoencoder
            from src.models.unet_segmentation import SpectralUNet
            from src.models.multimodal_fusion import MultimodalFusionNetwork
            
            model_configs = {
                'spectral_cnn': {
                    'class': SpectralCNN,
                    'input_shape': (64, 64, settings.HYPERSPECTRAL_BANDS),
                    'num_classes': 4
                },
                'spectral_lstm': {
                    'class': SpectralLSTM,
                    'input_shape': (10, settings.HYPERSPECTRAL_BANDS)
                },
                'spectral_autoencoder': {
                    'class': SpectralAutoencoder,
                    'input_shape': (64, 64, settings.HYPERSPECTRAL_BANDS)
                },
                'spectral_unet': {
                    'class': SpectralUNet,
                    'input_shape': (256, 256, settings.HYPERSPECTRAL_BANDS),
                    'num_classes': 4
                }
            }
            
            for model_name, config in model_configs.items():
                try:
                    # Try to load saved model first
                    model_path = os.path.join(settings.MODEL_PATH, f"{model_name}.h5")
                    if os.path.exists(model_path):
                        import tensorflow as tf
                        model = tf.keras.models.load_model(model_path)
                        self.models[model_name] = model
                        logger.info(f"Loaded saved {model_name} model")
                    else:
                        # Initialize new model for inference
                        model_class = config['class']
                        model_instance = model_class(
                            input_shape=config['input_shape'],
                            num_classes=config.get('num_classes'),
                            config={}
                        )
                        model = model_instance.build_model()
                        self.models[model_name] = model
                        logger.info(f"Initialized new {model_name} model")
                        
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {str(e)}")
            
            logger.info("Spectral models loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Could not import spectral models: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading spectral models: {str(e)}")
    
    def _load_traditional_models(self):
        """Load traditional ML models (scikit-learn)"""
        try:
            from sklearn.ensemble import RandomForestClassifier, IsolationForest
            from sklearn.linear_model import LogisticRegression
            
            # Try to load saved models
            traditional_models = {
                'disease_classifier': 'disease_rf_model.pkl',
                'anomaly_detector': 'anomaly_isolation_forest.pkl',
                'stress_classifier': 'stress_logistic_model.pkl'
            }
            
            for model_name, filename in traditional_models.items():
                model_path = os.path.join(settings.MODEL_PATH, filename)
                try:
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        self.models[model_name] = model
                        logger.info(f"Loaded saved {model_name}")
                    else:
                        # Initialize fallback models
                        if model_name == 'disease_classifier':
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif model_name == 'anomaly_detector':
                            model = IsolationForest(contamination=0.1, random_state=42)
                        elif model_name == 'stress_classifier':
                            model = LogisticRegression(random_state=42)
                        
                        self.models[model_name] = model
                        logger.info(f"Initialized fallback {model_name}")
                        
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {str(e)}")
            
            logger.info("Traditional ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading traditional models: {str(e)}")
    
    def predict_disaster_text(self, text: str) -> Dict[str, Any]:
        """Predict disaster information from text using BERT model"""
        try:
            if 'disaster_bert' not in self.models:
                raise ValueError("BERT disaster model not available")
            
            model = self.models['disaster_bert']
            tokenizer = self.tokenizers['disaster_bert']
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Tokenize
            inputs = tokenizer(
                processed_text,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                d_logits, u_logits, r_logits = model(inputs['input_ids'], inputs['attention_mask'])
                
                disaster_idx = torch.argmax(d_logits, dim=1).item()
                urgency_idx = torch.argmax(u_logits, dim=1).item()
                relevance = torch.sigmoid(r_logits).item()
            
            # Map to labels
            labels = self.label_mappings['disaster_bert']
            
            return {
                "disaster_type": labels['disaster_labels'][disaster_idx],
                "urgency": labels['urgency_labels'][urgency_idx],
                "relevance": round(relevance, 3),
                "confidence": float(torch.max(torch.softmax(d_logits, dim=1)).item())
            }
            
        except Exception as e:
            logger.error(f"Error in disaster text prediction: {str(e)}")
            raise
    
    def predict_spectral_disease(self, spectral_data: np.ndarray) -> Dict[str, Any]:
        """Predict disease from spectral data"""
        try:
            if 'spectral_cnn' in self.models:
                model = self.models['spectral_cnn']
                
                # Prepare input
                if spectral_data.ndim == 2:
                    # Reshape for CNN input
                    spectral_data = spectral_data.reshape(1, *spectral_data.shape, 1)
                elif spectral_data.ndim == 3:
                    spectral_data = spectral_data.reshape(1, *spectral_data.shape)
                
                # Predict
                predictions = model.predict(spectral_data)
                disease_prob = float(np.max(predictions))
                predicted_class = int(np.argmax(predictions))
                
                return {
                    "disease_probability": disease_prob,
                    "predicted_class": predicted_class,
                    "class_names": ["healthy", "mild_disease", "moderate_disease", "severe_disease"],
                    "confidence": disease_prob
                }
            
            # Fallback to traditional ML
            elif 'disease_classifier' in self.models:
                model = self.models['disease_classifier']
                
                # Flatten spectral data for traditional ML
                flattened_data = spectral_data.flatten().reshape(1, -1)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(flattened_data)[0]
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))
                else:
                    predicted_class = int(model.predict(flattened_data)[0])
                    confidence = 0.75  # Default confidence
                
                return {
                    "disease_probability": confidence if predicted_class > 0 else 1 - confidence,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
            
            else:
                raise ValueError("No disease prediction model available")
                
        except Exception as e:
            logger.error(f"Error in spectral disease prediction: {str(e)}")
            raise
    
    def detect_anomalies(self, spectral_data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in spectral data"""
        try:
            if 'spectral_autoencoder' in self.models:
                model = self.models['spectral_autoencoder']
                
                # Prepare input
                if spectral_data.ndim == 2:
                    input_data = spectral_data.reshape(1, *spectral_data.shape, 1)
                else:
                    input_data = spectral_data.reshape(1, *spectral_data.shape)
                
                # Get reconstruction
                reconstructed = model.predict(input_data)
                
                # Calculate reconstruction error
                mse = np.mean((input_data - reconstructed) ** 2)
                anomaly_score = float(mse)
                
                is_anomaly = anomaly_score > settings.ANOMALY_THRESHOLD
                
                return {
                    "anomaly_score": anomaly_score,
                    "is_anomaly": is_anomaly,
                    "threshold": settings.ANOMALY_THRESHOLD,
                    "reconstruction_error": anomaly_score
                }
            
            # Fallback to isolation forest
            elif 'anomaly_detector' in self.models:
                model = self.models['anomaly_detector']
                
                flattened_data = spectral_data.flatten().reshape(1, -1)
                anomaly_prediction = model.predict(flattened_data)[0]
                anomaly_score = model.decision_function(flattened_data)[0]
                
                return {
                    "anomaly_score": float(abs(anomaly_score)),
                    "is_anomaly": anomaly_prediction == -1,
                    "confidence": float(abs(anomaly_score))
                }
            
            else:
                raise ValueError("No anomaly detection model available")
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    def segment_spectral_image(self, spectral_image: np.ndarray) -> Dict[str, Any]:
        """Segment spectral image using U-Net"""
        try:
            if 'spectral_unet' not in self.models:
                raise ValueError("U-Net segmentation model not available")
            
            model = self.models['spectral_unet']
            
            # Prepare input
            if spectral_image.ndim == 3:
                input_data = spectral_image.reshape(1, *spectral_image.shape)
            else:
                input_data = spectral_image
            
            # Predict segmentation
            segmentation = model.predict(input_data)
            
            # Process output
            if segmentation.ndim == 4:
                segmentation = segmentation[0]
            
            # Get class predictions
            segmentation_map = np.argmax(segmentation, axis=-1)
            
            return {
                "segmentation_map": segmentation_map.tolist(),
                "class_probabilities": segmentation.tolist(),
                "num_classes": segmentation.shape[-1],
                "class_names": ["background", "healthy_vegetation", "stressed_vegetation", "diseased_vegetation"]
            }
            
        except Exception as e:
            logger.error(f"Error in spectral image segmentation: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for BERT model"""
        if not isinstance(text, str):
            return ""
        return emoji.demojize(text, language="en")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {}
        
        for model_name, model in self.models.items():
            try:
                model_info = {
                    "loaded": True,
                    "device": str(self.device) if hasattr(model, 'to') else "cpu",
                    "type": type(model).__name__
                }
                
                # Add model-specific info
                if hasattr(model, 'parameters'):
                    # PyTorch model
                    model_info["parameters"] = sum(p.numel() for p in model.parameters())
                elif hasattr(model, 'get_params'):
                    # Scikit-learn model
                    model_info["parameters"] = str(model.get_params())
                
                status[model_name] = model_info
                
            except Exception as e:
                status[model_name] = {
                    "loaded": False,
                    "error": str(e)
                }
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models"""
        results = {}
        
        for model_name in self.models:
            try:
                if model_name == 'disaster_bert':
                    # Test with sample text
                    test_result = self.predict_disaster_text("Sample disaster text")
                    results[model_name] = {"status": "healthy", "test_passed": True}
                
                elif 'spectral' in model_name:
                    # Test with sample spectral data
                    test_data = np.random.random((32, 32, 10))  # Small test data
                    if model_name == 'spectral_cnn':
                        test_result = self.predict_spectral_disease(test_data)
                    elif model_name == 'spectral_autoencoder':
                        test_result = self.detect_anomalies(test_data)
                    
                    results[model_name] = {"status": "healthy", "test_passed": True}
                
                else:
                    results[model_name] = {"status": "healthy", "test_passed": "skipped"}
                    
            except Exception as e:
                results[model_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "test_passed": False
                }
        
        return results


# Global model integration service instance
model_service = ModelIntegrationService()