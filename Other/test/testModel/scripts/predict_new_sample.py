"""
Make predictions on new hyperspectral image and sensor input data using the trained hybrid model.
This script combines 3D CNN+LSTM spatial features with tabular sensor data.
"""

import numpy as np
import pandas as pd
import joblib
import os
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

class CropHealthPredictor:
    def __init__(self, models_path='../models'):
        self.models_path = models_path
        self.cnn_lstm_model = None
        self.rf_model = None
        self.label_encoder = None
        self.scaler = None
        self.image_shape = (64, 64, 50, 1)  # Hyperspectral image shape
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessors."""
        try:
            # Load CNN+LSTM model
            cnn_lstm_path = os.path.join(self.models_path, 'hybrid_cnn_lstm_model.h5')
            if os.path.exists(cnn_lstm_path):
                self.cnn_lstm_model = load_model(cnn_lstm_path)
                print("CNN+LSTM model loaded successfully.")
            else:
                print("Warning: CNN+LSTM model not found.")
            
            # Load Random Forest model
            rf_path = os.path.join(self.models_path, 'rf_classifier.pkl')
            self.rf_model = joblib.load(rf_path)
            print("Random Forest model loaded successfully.")
            
            # Load label encoder
            encoder_path = os.path.join(self.models_path, 'label_encoder.pkl')
            self.label_encoder = joblib.load(encoder_path)
            print("Label encoder loaded successfully.")
            
            # Load scaler
            scaler_path = os.path.join(self.models_path, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully.")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure models are trained first by running train_hybrid_model.py")
    
    def calculate_spectral_indices(self, image_cube):
        """Calculate spectral indices from hyperspectral image cube."""
        if len(image_cube.shape) == 4:
            image_cube = image_cube[0]  # Remove batch dimension
        
        # Remove channel dimension if present
        if image_cube.shape[-1] == 1:
            image_cube = image_cube[:, :, :, 0]
        
        # Select approximate bands (adjust based on your hyperspectral data)
        red_band = image_cube[:, :, 30] if image_cube.shape[2] > 30 else image_cube[:, :, -5]
        nir_band = image_cube[:, :, 40] if image_cube.shape[2] > 40 else image_cube[:, :, -3]
        swir_band = image_cube[:, :, 45] if image_cube.shape[2] > 45 else image_cube[:, :, -1]
        green_band = image_cube[:, :, 20] if image_cube.shape[2] > 20 else image_cube[:, :, 5]
        blue_band = image_cube[:, :, 10] if image_cube.shape[2] > 10 else image_cube[:, :, 0]
        
        # Calculate indices with epsilon to avoid division by zero
        epsilon = 1e-6
        
        ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
        ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)
        ndsi = (green_band - swir_band) / (green_band + swir_band + epsilon)
        chi = (nir_band - red_band) / (nir_band + red_band + epsilon)  # Same as NDVI for this example
        gndvi = (nir_band - green_band) / (nir_band + green_band + epsilon)
        
        return {
            "NDVI": np.nanmean(ndvi),
            "NDWI": np.nanmean(ndwi),
            "NDSI": np.nanmean(ndsi),
            "CHI": np.nanmean(chi),
            "GNDVI": np.nanmean(gndvi)
        }
    
    def _resize_hyperspectral_image(self, img):
        """Resize hyperspectral image to target shape."""
        h, w, bands, channels = img.shape
        th, tw, tb, tc = self.image_shape
        
        # Resize spatial dimensions
        if h >= th and w >= tw:
            start_h = (h - th) // 2
            start_w = (w - tw) // 2
            img_resized = img[start_h:start_h+th, start_w:start_w+tw, :, :]
        else:
            result = np.zeros((th, tw, bands, channels))
            end_h = min(h, th)
            end_w = min(w, tw)
            result[:end_h, :end_w, :, :] = img[:end_h, :end_w, :, :]
            img_resized = result
        
        # Handle spectral bands
        if bands >= tb:
            img_resized = img_resized[:, :, :tb, :]
        else:
            result = np.zeros(self.image_shape)
            result[:, :, :bands, :] = img_resized
            img_resized = result
        
        return img_resized
    
    def extract_spatial_temporal_features(self, image_sequence):
        """Extract spatial-temporal features using CNN+LSTM model."""
        if self.cnn_lstm_model is not None:
            features = self.cnn_lstm_model.predict(image_sequence, verbose=0)
            return features[0]  # Remove batch dimension
        else:
            # Simulate spatial-temporal features if model not available
            print("Simulating spatial-temporal features...")
            return np.random.rand(32)  # 32 features as defined in model architecture
    
    def prepare_tabular_features(self, sensor_data, spectral_indices=None):
        """Prepare tabular sensor data and spectral indices for prediction."""
        feature_columns = [
            'NDVI', 'NDWI', 'NDSI', 'CHI', 'GNDVI',
            'SoilMoisture', 'AirTemp', 'Humidity', 'Rainfall', 'pH'
        ]
        
        # Merge sensor data with spectral indices
        all_features = {**sensor_data}
        if spectral_indices:
            all_features.update(spectral_indices)
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in all_features:
                print(f"Warning: {col} not found. Using default value.")
                if col in ['NDVI', 'NDWI', 'NDSI', 'CHI', 'GNDVI']:
                    all_features[col] = 0.5  # Default spectral index value
                else:
                    all_features[col] = 0.0  # Default sensor value
        
        # Create DataFrame and select features
        df = pd.DataFrame([all_features])
        X_tabular = df[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X_tabular)
        
        return X_scaled[0]  # Return single sample
    
    def predict(self, image_path=None, sensor_data=None):
        """Make prediction using hybrid model."""
        predictions = {}
        
        # Process hyperspectral image if provided
        if image_path and self.cnn_lstm_model:
            try:
                # Load and preprocess image
                image_data = self.preprocess_hyperspectral_image(image_path)
                
                # Calculate spectral indices from the image
                spectral_indices = self.calculate_spectral_indices(image_data)
                
                # Prepare tabular features (combining sensor data with spectral indices)
                if sensor_data:
                    tabular_features = self.prepare_tabular_features(sensor_data, spectral_indices)
                else:
                    # Use only spectral indices with default sensor values
                    default_sensor = {
                        'SoilMoisture': 0.3, 'AirTemp': 25.0, 'Humidity': 65.0, 
                        'Rainfall': 2.5, 'pH': 6.5
                    }
                    tabular_features = self.prepare_tabular_features(default_sensor, spectral_indices)
                
                # Prepare inputs for CNN+LSTM model
                tabular_input = np.expand_dims(tabular_features, -1)  # Add sequence dimension
                tabular_input = np.expand_dims(tabular_input, 0)      # Add batch dimension
                
                # Make CNN+LSTM prediction
                cnn_pred = self.cnn_lstm_model.predict([image_data, tabular_input], verbose=0)
                cnn_pred_class = np.argmax(cnn_pred, axis=1)[0]
                cnn_confidence = float(np.max(cnn_pred))
                
                predictions['cnn_lstm'] = {
                    'predicted_class': self.label_encoder.inverse_transform([cnn_pred_class])[0],
                    'confidence': cnn_confidence,
                    'class_probabilities': {
                        self.label_encoder.classes_[i]: float(prob) 
                        for i, prob in enumerate(cnn_pred[0])
                    },
                    'spectral_indices': spectral_indices
                }
                
                # Extract features for Random Forest ensemble
                if self.rf_model:
                    feature_extractor = Model(
                        inputs=self.cnn_lstm_model.inputs,
                        outputs=self.cnn_lstm_model.layers[-2].output
                    )
                    deep_features = feature_extractor.predict([image_data, tabular_input], verbose=0)
                    
                    # Random Forest prediction on deep features
                    rf_pred_proba = self.rf_model.predict_proba(deep_features)[0]
                    rf_pred_class = self.rf_model.predict(deep_features)[0]
                    
                    predictions['random_forest'] = {
                        'predicted_class': self.label_encoder.inverse_transform([rf_pred_class])[0],
                        'confidence': float(np.max(rf_pred_proba)),
                        'class_probabilities': {
                            self.label_encoder.classes_[i]: float(prob) 
                            for i, prob in enumerate(rf_pred_proba)
                        }
                    }
                    
                    # Ensemble prediction (weighted average)
                    ensemble_proba = 0.6 * cnn_pred[0] + 0.4 * rf_pred_proba
                    ensemble_class = np.argmax(ensemble_proba)
                    
                    predictions['ensemble'] = {
                        'predicted_class': self.label_encoder.inverse_transform([ensemble_class])[0],
                        'confidence': float(np.max(ensemble_proba)),
                        'class_probabilities': {
                            self.label_encoder.classes_[i]: float(prob) 
                            for i, prob in enumerate(ensemble_proba)
                        }
                    }
                
            except Exception as e:
                print(f"Error processing image: {e}")
                predictions['error'] = str(e)
        
        # Add metadata
        predictions['prediction_timestamp'] = datetime.now().isoformat()
        predictions['model_info'] = {
            'cnn_lstm_available': self.cnn_lstm_model is not None,
            'rf_available': self.rf_model is not None,
            'image_processed': image_path is not None
        }
        
        return predictions
    
    def predict_from_csv(self, csv_path, roi_id=None, date=None):
        """Make prediction using data from CSV file."""
        df = pd.read_csv(csv_path)
        
        if roi_id and date:
            # Filter for specific ROI and date
            mask = (df['roi_id'] == roi_id) & (df['date'] == date)
            filtered_df = df[mask]
            
            if filtered_df.empty:
                print(f"No data found for ROI {roi_id} on date {date}")
                return None
            
            sample = filtered_df.iloc[0]
        else:
            # Use first sample
            sample = df.iloc[0]
            roi_id = sample['roi_id']
            date = sample['date']
        
        print(f"Making prediction for ROI: {roi_id}, Date: {date}")
        
        # Convert sample to dictionary (excluding target variables)
        sensor_data = sample.drop(['health_score', 'health_class'], errors='ignore').to_dict()
        
        # Make prediction
        result = self.predict(sensor_data=sensor_data)
        
        # Add ground truth if available
        if 'health_class' in sample:
            result['ground_truth'] = {
                'actual_class': sample['health_class'],
                'actual_score': sample.get('health_score', 'N/A')
            }
        
        return result

def main():
    """Main prediction function with command line interface."""
    parser = argparse.ArgumentParser(description='Predict crop health using hybrid model')
    parser.add_argument('--csv', type=str, help='Path to CSV file with sensor data')
    parser.add_argument('--roi_id', type=str, help='ROI ID to predict')
    parser.add_argument('--date', type=str, help='Date to predict (YYYY-MM-DD)')
    parser.add_argument('--image', type=str, help='Path to hyperspectral image file (.npy)')
    parser.add_argument('--models_path', type=str, default='../models', help='Path to saved models')
    parser.add_argument('--soil_moisture', type=float, default=0.3, help='Soil moisture (0-1)')
    parser.add_argument('--air_temp', type=float, default=25.0, help='Air temperature (°C)')
    parser.add_argument('--humidity', type=float, default=65.0, help='Humidity (%)')
    parser.add_argument('--rainfall', type=float, default=2.5, help='Rainfall (mm)')
    parser.add_argument('--ph', type=float, default=6.5, help='Soil pH')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CropHealthPredictor(models_path=args.models_path)
    
    if args.csv:
        # Predict from CSV
        result = predictor.predict_from_csv(args.csv, args.roi_id, args.date)
    elif args.image:
        # Manual prediction with image and sensor data
        sensor_data = {
            'SoilMoisture': args.soil_moisture,
            'AirTemp': args.air_temp,
            'Humidity': args.humidity,
            'Rainfall': args.rainfall,
            'pH': args.ph
        }
        result = predictor.predict(image_path=args.image, sensor_data=sensor_data)
    else:
        print("Please provide either --csv or --image argument.")
        print("Examples:")
        print("  python predict_new_sample.py --csv ../data/final_dataset.csv --roi_id roi_01")
        print("  python predict_new_sample.py --image ../data/images/roi_01_20210714.npy")
        return
    
    if result:
        print("\n" + "="*50)
        print("CROP HEALTH PREDICTION RESULTS")
        print("="*50)
        
        if 'ground_truth' in result:
            print(f"Ground Truth: {result['ground_truth']['actual_class']}")
            if result['ground_truth']['actual_score'] != 'N/A':
                print(f"Actual Score: {result['ground_truth']['actual_score']:.2f}")
            print("-" * 30)
        
        if 'ensemble' in result:
            print(f"Ensemble Prediction: {result['ensemble']['predicted_class']}")
            print(f"Ensemble Confidence: {result['ensemble']['confidence']:.3f}")
            print("\nEnsemble Class Probabilities:")
            for class_name, prob in result['ensemble']['class_probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
        
        if 'cnn_lstm' in result:
            print(f"\nCNN+LSTM Prediction: {result['cnn_lstm']['predicted_class']}")
            print(f"CNN+LSTM Confidence: {result['cnn_lstm']['confidence']:.3f}")
            
            if 'spectral_indices' in result['cnn_lstm']:
                print("\nCalculated Spectral Indices:")
                for idx, value in result['cnn_lstm']['spectral_indices'].items():
                    print(f"  {idx}: {value:.3f}")
        
        if 'random_forest' in result:
            print(f"\nRandom Forest Prediction: {result['random_forest']['predicted_class']}")
            print(f"Random Forest Confidence: {result['random_forest']['confidence']:.3f}")
        
        print("\nModel Information:")
        if 'model_info' in result:
            info = result['model_info']
            print(f"  CNN+LSTM Available: {info['cnn_lstm_available']}")
            print(f"  Random Forest Available: {info['rf_available']}")
            print(f"  Image Processed: {info['image_processed']}")
        
        print("="*50)

if __name__ == "__main__":
    main()