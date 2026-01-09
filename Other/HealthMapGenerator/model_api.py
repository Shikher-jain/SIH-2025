"""
Flask API to serve trained PyTorch model (.pth) for health map predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import os
import json
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
device = None
transform = None
model_config = {}

class HealthMapModel(nn.Module):
    """
    Example model architecture - modify based on your actual model
    This is a sample CNN for image classification/regression
    """
    def __init__(self, num_classes=5):  # 5 health levels
        super(HealthMapModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model(model_path, model_type="classification", num_classes=5):
    """
    Load the trained PyTorch model
    """
    global model, device, transform, model_config
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model architecture (modify based on your model)
        if model_type == "classification":
            model = HealthMapModel(num_classes=num_classes)
        else:
            # For regression or other tasks
            model = HealthMapModel(num_classes=1)
        
        # Load the trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    model_config = checkpoint.get('config', {})
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model_config = checkpoint.get('config', {})
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            logger.info(f"✅ Model loaded successfully from {model_path}")
            
            # Define image preprocessing transforms
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            return True
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_data, input_format="base64"):
    """
    Preprocess image for model input
    """
    try:
        if input_format == "base64":
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif input_format == "file":
            # Load from file path
            image = Image.open(image_data)
        else:
            return None
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if transform:
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(device)
        
        return None
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_health_level(image_tensor):
    """
    Predict health level from image tensor
    """
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # For classification
            if outputs.shape[1] > 1:
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
                
                # Map class to health level
                health_levels = {
                    0: "Very Poor",
                    1: "Poor", 
                    2: "Moderate",
                    3: "Good",
                    4: "Excellent"
                }
                
                return {
                    "health_level": health_levels.get(predicted_class, "Unknown"),
                    "class_id": predicted_class,
                    "confidence": float(confidence),
                    "probabilities": [float(p) for p in probabilities.squeeze().tolist()]
                }
            else:
                # For regression
                health_score = outputs.item()
                normalized_score = max(0, min(1, (health_score + 1) / 2))  # Normalize to [0,1]
                
                return {
                    "health_score": float(health_score),
                    "normalized_score": float(normalized_score)
                }
                
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized",
        "model_config": model_config,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expected JSON format:
    {
        "image": "base64_encoded_image_string",
        "format": "base64"  # or "file"
    }
    """
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please load a model first using /load_model endpoint"
            }), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Please provide 'image' field in JSON"
            }), 400
        
        image_data = data['image']
        input_format = data.get('format', 'base64')
        
        # Preprocess image
        image_tensor = preprocess_image(image_data, input_format)
        if image_tensor is None:
            return jsonify({
                "error": "Image preprocessing failed",
                "message": "Could not process the provided image"
            }), 400
        
        # Make prediction
        prediction = predict_health_level(image_tensor)
        if prediction is None:
            return jsonify({
                "error": "Prediction failed",
                "message": "Model prediction failed"
            }), 500
        
        response = {
            "success": True,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "device": str(device),
                "input_shape": list(image_tensor.shape)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """
    Load model endpoint
    Expected JSON format:
    {
        "model_path": "path/to/model.pth",
        "model_type": "classification",  # or "regression"
        "num_classes": 5
    }
    """
    try:
        data = request.get_json()
        if not data or 'model_path' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Please provide 'model_path' field in JSON"
            }), 400
        
        model_path = data['model_path']
        model_type = data.get('model_type', 'classification')
        num_classes = data.get('num_classes', 5)
        
        success = load_model(model_path, model_type, num_classes)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Model loaded successfully from {model_path}",
                "model_info": {
                    "path": model_path,
                    "type": model_type,
                    "num_classes": num_classes,
                    "device": str(device)
                }
            })
        else:
            return jsonify({
                "error": "Model loading failed",
                "message": f"Could not load model from {model_path}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in load_model endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Expected JSON format:
    {
        "images": ["base64_1", "base64_2", ...],
        "format": "base64"
    }
    """
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please load a model first using /load_model endpoint"
            }), 500
        
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Please provide 'images' field in JSON"
            }), 400
        
        images = data['images']
        input_format = data.get('format', 'base64')
        
        if not isinstance(images, list):
            return jsonify({
                "error": "Invalid format",
                "message": "'images' must be a list"
            }), 400
        
        results = []
        
        for i, image_data in enumerate(images):
            try:
                # Preprocess image
                image_tensor = preprocess_image(image_data, input_format)
                if image_tensor is None:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "Image preprocessing failed"
                    })
                    continue
                
                # Make prediction
                prediction = predict_health_level(image_tensor)
                if prediction is None:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "Prediction failed"
                    })
                    continue
                
                results.append({
                    "index": i,
                    "success": True,
                    "prediction": prediction
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        response = {
            "success": True,
            "results": results,
            "total_processed": len(images),
            "successful_predictions": sum(1 for r in results if r["success"]),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            "model_loaded": False,
            "message": "No model loaded"
        })
    
    try:
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return jsonify({
            "model_loaded": True,
            "device": str(device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_config": model_config,
            "input_shape": [3, 256, 256],  # Expected input shape
            "transform_info": {
                "resize": [256, 256],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225]
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": "Could not get model info",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Auto-load model if model.pth exists
    default_model_path = './model.pth'
    if os.path.exists(default_model_path):
        logger.info(f"Auto-loading model from {default_model_path}")
        load_model(default_model_path)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)