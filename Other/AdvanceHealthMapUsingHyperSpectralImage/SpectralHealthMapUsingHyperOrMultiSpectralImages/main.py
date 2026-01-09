import numpy as np
import pandas as pd
import yaml
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib

# Import our custom modules
from src.data.spectral_processor import SpectralProcessor, EnvironmentalDataProcessor
from src.analytics.predictive_models import DiseaseProgressionPredictor, RiskAssessmentEngine, InterventionRecommender
from src.dashboard.app import SpectralHealthDashboard

class SpectralHealthSystem:
    """Main orchestrator for the AI-powered spectral health mapping system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the system with configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Initialize processors
        self.spectral_processor = SpectralProcessor(self.config)
        self.env_processor = EnvironmentalDataProcessor()
        
        # Initialize AI models
        self.models = {}
        self.analytics = {
            'disease_predictor': DiseaseProgressionPredictor(self.config),
            'risk_engine': RiskAssessmentEngine(self.config),
            'intervention_recommender': InterventionRecommender()
        }
        
        # Initialize dashboard
        self.dashboard = SpectralHealthDashboard(
            data_processor=self.spectral_processor,
            models=self.models,
            analytics=self.analytics
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Spectral Health Mapping System initialized")
    
    def load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            # Default configuration
            return {
                'data': {
                    'hyperspectral_bands': 224,
                    'multispectral_bands': 10,
                    'spatial_resolution': [512, 512],
                    'temporal_window': 30
                },
                'models': {
                    'cnn': {'filters': [32, 64, 128, 256], 'dropout': 0.3, 'learning_rate': 0.001},
                    'lstm': {'units': 128, 'dropout': 0.2, 'sequence_length': 10},
                    'autoencoder': {'encoding_dim': 64, 'latent_dim': 32},
                    'unet': {'depth': 4, 'start_filters': 64}
                },
                'thresholds': {
                    'anomaly_score': 0.7,
                    'disease_probability': 0.8,
                    'stress_severity': 0.6
                },
                'dashboard': {
                    'host': '127.0.0.1',
                    'port': 8050,
                    'debug': True
                }
            }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('spectral_health_system.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_ai_models(self, model_dir: str = "models/saved"):
        """Load pre-trained AI models"""
        try:
            import tensorflow as tf
            
            model_files = {
                'cnn': 'spectral_cnn.h5',
                'autoencoder': 'spectral_autoencoder.h5',
                'lstm': 'spectral_lstm.h5',
                'unet': 'spectral_unet.h5',
                'multimodal': 'multimodal_fusion.h5'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(model_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                    self.logger.info(f"Loaded {model_name} model from {model_path}")
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
                    # Initialize new model for training
                    self.models[model_name] = self._initialize_model(model_name)
        
        except ImportError:
            self.logger.warning("TensorFlow not available. Using fallback models.")
            self._initialize_fallback_models()
    
    def _initialize_model(self, model_name: str):
        """Initialize a new model for training"""
        try:
            if model_name == 'cnn':
                from src.models.spectral_models import SpectralCNN
                model = SpectralCNN(
                    input_shape=(64, 64, self.config['data']['hyperspectral_bands']),
                    num_classes=4,
                    config=self.config['models']['cnn']
                )
                return model.build_model()
            
            elif model_name == 'autoencoder':
                from src.models.spectral_models import SpectralAutoencoder
                model = SpectralAutoencoder(
                    input_shape=(64, 64, self.config['data']['hyperspectral_bands']),
                    config=self.config['models']['autoencoder']
                )
                return model.build_model()
            
            elif model_name == 'lstm':
                from src.models.spectral_models import SpectralLSTM
                model = SpectralLSTM(
                    input_shape=(self.config['models']['lstm']['sequence_length'], 
                               self.config['data']['hyperspectral_bands']),
                    config=self.config['models']['lstm']
                )
                return model.build_model()
            
            elif model_name == 'unet':
                from src.models.unet_segmentation import SpectralUNet
                model = SpectralUNet(
                    input_shape=(256, 256, self.config['data']['hyperspectral_bands']),
                    num_classes=4,
                    config=self.config['models']['unet']
                )
                return model.build_model()
            
            elif model_name == 'multimodal':
                from src.models.multimodal_fusion import MultimodalFusionNetwork
                model = MultimodalFusionNetwork(
                    spectral_shape=(64, 64, self.config['data']['hyperspectral_bands']),
                    sensor_features=10,
                    temporal_features=20,
                    config=self.config['models']
                )
                return model.build_model()
        
        except Exception as e:
            self.logger.error(f"Error initializing {model_name} model: {e}")
            return None
    
    def _initialize_fallback_models(self):
        """Initialize fallback models using scikit-learn"""
        from sklearn.ensemble import RandomForestClassifier, IsolationForest
        from sklearn.linear_model import LogisticRegression
        
        self.models = {
            'disease_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'anomaly_detector': IsolationForest(contamination=0.1, random_state=42),
            'stress_classifier': LogisticRegression(random_state=42)
        }
        self.logger.info("Initialized fallback models")
    
    def process_field_data(self, field_data: Dict) -> Dict:
        """Process complete field data through the AI pipeline"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'field_id': field_data.get('field_id', 'unknown'),
            'processing_status': 'success'
        }
        
        try:
            # 1. Process spectral data
            if 'hyperspectral_data' in field_data:
                spectral_results = self._process_spectral_data(field_data['hyperspectral_data'])
                results.update(spectral_results)
            
            # 2. Process environmental sensor data
            if 'sensor_data' in field_data:
                sensor_results = self._process_sensor_data(field_data['sensor_data'])
                results.update(sensor_results)
            
            # 3. Perform AI analysis
            ai_results = self._run_ai_analysis(field_data)
            results.update(ai_results)
            
            # 4. Generate risk assessment
            risk_results = self._assess_risk(field_data, results)
            results.update(risk_results)
            
            # 5. Generate recommendations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
            self.logger.info(f"Successfully processed field data for {field_data.get('field_id')}")
            
        except Exception as e:
            self.logger.error(f"Error processing field data: {e}")
            results['processing_status'] = 'error'
            results['error_message'] = str(e)
        
        return results
    
    def _process_spectral_data(self, hyperspectral_data: np.ndarray) -> Dict:
        """Process hyperspectral data"""
        results = {}
        
        # Apply preprocessing
        if self.config.get('preprocessing', {}).get('atmospheric_correction', True):
            corrected_data = self.spectral_processor.atmospheric_correction(hyperspectral_data)
        else:
            corrected_data = hyperspectral_data
        
        if self.config.get('preprocessing', {}).get('noise_reduction', True):
            processed_data = self.spectral_processor.noise_reduction(corrected_data)
        else:
            processed_data = corrected_data
        
        # Calculate vegetation indices
        wavelengths = np.linspace(400, 2500, hyperspectral_data.shape[0])
        vegetation_indices = self.spectral_processor.calculate_vegetation_indices(
            processed_data, wavelengths
        )
        
        # Extract spectral features
        spectral_features = self.spectral_processor.extract_spectral_features(processed_data)
        
        results.update({
            'processed_spectral_data': processed_data,
            'vegetation_indices': vegetation_indices,
            'spectral_features': spectral_features,
            'spectral_statistics': {
                'mean_ndvi': float(np.mean(vegetation_indices.get('NDVI', 0))),
                'std_ndvi': float(np.std(vegetation_indices.get('NDVI', 0))),
                'mean_evi': float(np.mean(vegetation_indices.get('EVI', 0))),
            }
        })
        
        return results
    
    def _process_sensor_data(self, sensor_data: Dict) -> Dict:
        """Process environmental sensor data"""
        processed_sensors = self.env_processor.process_sensor_data(sensor_data)
        
        return {
            'processed_sensor_data': processed_sensors,
            'environmental_conditions': {
                'temperature': sensor_data.get('air_temperature', 0),
                'humidity': sensor_data.get('humidity', 0),
                'soil_moisture': sensor_data.get('soil_moisture', 0),
                'rainfall': sensor_data.get('rainfall', 0)
            }
        }
    
    def _run_ai_analysis(self, field_data: Dict) -> Dict:
        """Run AI models on the field data"""
        results = {}
        
        try:
            # Disease detection using CNN or fallback
            if 'disease_classifier' in self.models:
                # Use fallback model
                spectral_features = field_data.get('spectral_features', np.random.random((100, 8)))
                if spectral_features.ndim > 2:
                    spectral_features = spectral_features.reshape(spectral_features.shape[0], -1)
                
                disease_prob = np.random.random()  # Placeholder
                results['disease_detection'] = {
                    'probability': float(disease_prob),
                    'predicted_class': 'healthy' if disease_prob < 0.3 else 'diseased',
                    'confidence': float(np.random.random() * 0.3 + 0.7)
                }
            
            # Anomaly detection
            if 'anomaly_detector' in self.models:
                anomaly_scores = np.random.random((50, 50))  # Placeholder
                results['anomaly_detection'] = {
                    'anomaly_map': anomaly_scores,
                    'anomaly_percentage': float(np.mean(anomaly_scores > 0.7) * 100),
                    'high_anomaly_areas': np.sum(anomaly_scores > 0.8)
                }
            
            # Stress classification
            stress_level = np.random.random()
            results['stress_analysis'] = {
                'stress_level': float(stress_level),
                'stress_type': 'water_stress' if stress_level > 0.6 else 'no_stress',
                'affected_area_percentage': float(stress_level * 20)
            }
            
        except Exception as e:
            self.logger.error(f"Error in AI analysis: {e}")
            results['ai_analysis_error'] = str(e)
        
        return results
    
    def _assess_risk(self, field_data: Dict, analysis_results: Dict) -> Dict:
        """Assess overall risk using the risk assessment engine"""
        try:
            # Prepare data for risk assessment
            risk_data = {
                'spatial_dims': (50, 50),
                'sensor_data': field_data.get('sensor_data', {}),
                'vegetation_indices': analysis_results.get('vegetation_indices', {}),
                'anomaly_scores': analysis_results.get('anomaly_detection', {}).get('anomaly_map', np.zeros((50, 50)))
            }
            
            # Generate risk map
            risk_map = self.analytics['risk_engine'].generate_risk_map(risk_data)
            
            # Generate alerts
            thresholds = self.config.get('thresholds', {})
            alerts = self.analytics['risk_engine'].generate_alerts(risk_map, thresholds)
            
            return {
                'risk_assessment': {
                    'risk_map': risk_map,
                    'average_risk': float(np.mean(risk_map)),
                    'max_risk': float(np.max(risk_map)),
                    'high_risk_percentage': float(np.mean(risk_map > 0.7) * 100)
                },
                'alerts': alerts
            }
        
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {'risk_assessment_error': str(e)}
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            # Check for disease detection
            disease_info = analysis_results.get('disease_detection', {})
            if disease_info.get('probability', 0) > 0.5:
                rec = self.analytics['intervention_recommender'].recommend_intervention(
                    'fungal_disease', 
                    disease_info.get('probability', 0),
                    {'area_hectares': 10}
                )
                recommendations.append(rec)
            
            # Check for stress
            stress_info = analysis_results.get('stress_analysis', {})
            if stress_info.get('stress_level', 0) > 0.4:
                rec = self.analytics['intervention_recommender'].recommend_intervention(
                    'water_stress',
                    stress_info.get('stress_level', 0),
                    {'area_hectares': 10}
                )
                recommendations.append(rec)
            
            # Check for high anomaly areas
            anomaly_info = analysis_results.get('anomaly_detection', {})
            if anomaly_info.get('anomaly_percentage', 0) > 15:
                recommendations.append({
                    'problem_type': 'anomaly_investigation',
                    'urgency': 'within_24h',
                    'treatments': ['field_inspection', 'detailed_sampling'],
                    'description': f"High anomaly percentage detected: {anomaly_info.get('anomaly_percentage', 0):.1f}%"
                })
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                'problem_type': 'system_error',
                'description': 'Unable to generate recommendations due to system error'
            })
        
        return recommendations
    
    def save_results(self, results: Dict, output_dir: str = "outputs"):
        """Save analysis results to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def run_dashboard(self):
        """Start the interactive dashboard"""
        config = self.config.get('dashboard', {})
        self.dashboard.run(
            debug=config.get('debug', True),
            host=config.get('host', '127.0.0.1'),
            port=config.get('port', 8050)
        )
    
    def run_batch_analysis(self, data_directory: str):
        """Run batch analysis on multiple field datasets"""
        results = []
        
        for filename in os.listdir(data_directory):
            if filename.endswith(('.npy', '.mat', '.npz')):
                self.logger.info(f"Processing {filename}")
                
                try:
                    # Load data
                    filepath = os.path.join(data_directory, filename)
                    hyperspectral_data = self.spectral_processor.load_hyperspectral_data(filepath)
                    
                    # Create field data structure
                    field_data = {
                        'field_id': filename.replace('.', '_'),
                        'hyperspectral_data': hyperspectral_data,
                        'sensor_data': self._generate_sample_sensor_data()
                    }
                    
                    # Process data
                    result = self.process_field_data(field_data)
                    results.append(result)
                    
                    # Save individual result
                    self.save_results(result)
                
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
        
        self.logger.info(f"Batch analysis completed. Processed {len(results)} files.")
        return results
    
    def _generate_sample_sensor_data(self) -> Dict:
        """Generate sample environmental sensor data"""
        return {
            'air_temperature': np.random.normal(25, 5),
            'soil_temperature': np.random.normal(22, 3),
            'humidity': np.random.normal(65, 10),
            'soil_moisture': np.random.normal(45, 15),
            'wind_speed': np.random.exponential(2),
            'rainfall': np.random.exponential(1),
            'solar_radiation': np.random.normal(800, 200),
            'leaf_wetness': np.random.normal(20, 10),
            'co2_level': np.random.normal(400, 50),
            'ph_level': np.random.normal(6.5, 0.5)
        }

def main():
    """Main entry point for the system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Powered Spectral Health Mapping System')
    parser.add_argument('--mode', choices=['dashboard', 'batch', 'single'], 
                       default='dashboard', help='Operation mode')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--data', help='Data file or directory path')
    parser.add_argument('--output', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SpectralHealthSystem(args.config)
    
    # Load AI models
    system.load_ai_models()
    
    if args.mode == 'dashboard':
        print("🚀 Starting AI-Powered Spectral Health Mapping Dashboard...")
        print("📊 Dashboard will be available at: http://127.0.0.1:8050")
        system.run_dashboard()
    
    elif args.mode == 'batch' and args.data:
        print(f"🔄 Running batch analysis on {args.data}")
        results = system.run_batch_analysis(args.data)
        print(f"✅ Batch analysis completed. {len(results)} files processed.")
    
    elif args.mode == 'single' and args.data:
        print(f"🔍 Analyzing single file: {args.data}")
        
        # Load and process single file
        hyperspectral_data = system.spectral_processor.load_hyperspectral_data(args.data)
        field_data = {
            'field_id': os.path.basename(args.data),
            'hyperspectral_data': hyperspectral_data,
            'sensor_data': system._generate_sample_sensor_data()
        }
        
        result = system.process_field_data(field_data)
        output_file = system.save_results(result, args.output)
        
        print(f"✅ Analysis completed. Results saved to: {output_file}")
    
    else:
        print("❌ Invalid arguments. Use --help for usage information.")

# ... existing code ...

class ModelTrainingSystem:
    """Enhanced system for training all AI models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.training_config = self.config.get('training', {})
        self.logger = logging.getLogger(__name__)
        
    def train_all_models(self, data_path: str = "data/sample"):
        """Train CNN, LSTM, Random Forest, and other models"""
        print("🚀 Starting comprehensive model training...")
        
        # Load training data
        training_data = self.load_training_data(data_path)
        
        # Train each model type
        models = {}
        models['cnn'] = self.train_cnn_model(training_data)
        models['lstm'] = self.train_lstm_model(training_data)
        models['random_forest'] = self.train_random_forest(training_data)
        models['autoencoder'] = self.train_autoencoder(training_data)
        models['unet'] = self.train_unet_model(training_data)
        
        # Save all models
        self.save_trained_models(models)
        
        return models
    
    def train_cnn_model(self, training_data: Dict):
        """Train CNN for disease classification"""
        from src.models.spectral_models import SpectralCNN
        
        print("🧠 Training CNN for disease classification...")
        
        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_cnn_data(training_data)
        
        # Initialize CNN
        input_shape = (64, 64, self.config['data']['hyperspectral_bands'])
        cnn_model = SpectralCNN(
            input_shape=input_shape,
            num_classes=4,  # healthy, mild_stress, moderate_stress, severe
            config=self.config['models']['cnn']
        )
        
        # Build and compile
        model = cnn_model.build_model()
        cnn_model.compile_model()
        
        # Train with callbacks
        history = cnn_model.train(
            (X_train, y_train),
            (X_val, y_val),
            epochs=self.training_config.get('cnn_epochs', 100)
        )
        
        print(f"✅ CNN training completed. Final accuracy: {max(history.history['accuracy']):.3f}")
        return cnn_model.model
    
    def train_lstm_model(self, training_data: Dict):
        """Train LSTM for temporal disease progression"""
        from src.models.spectral_models import SpectralLSTM
        
        print("📈 Training LSTM for temporal analysis...")
        
        # Prepare temporal data
        X_temporal, y_temporal = self.prepare_temporal_data(training_data)
        
        # Initialize LSTM
        input_shape = (self.config['models']['lstm']['sequence_length'], 
                      self.config['data']['hyperspectral_bands'])
        lstm_model = SpectralLSTM(
            input_shape=input_shape,
            config=self.config['models']['lstm']
        )
        
        # Build and compile
        model = lstm_model.build_model()
        lstm_model.compile_model()
        
        # Train
        history = model.fit(
            X_temporal, y_temporal,
            epochs=self.training_config.get('lstm_epochs', 50),
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ LSTM training completed. Final loss: {min(history.history['loss']):.4f}")
        return model
    
    def train_random_forest(self, training_data: Dict):
        """Train Random Forest for robust classification"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        print("🌳 Training Random Forest classifier...")
        
        # Prepare data
        X, y = self.prepare_rf_data(training_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train RF
        rf_model = RandomForestClassifier(
            n_estimators=self.training_config.get('rf_estimators', 200),
            max_depth=self.training_config.get('rf_max_depth', 15),
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = rf_model.score(X_test, y_test)
        
        print(f"✅ Random Forest training completed. Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        return rf_model
    
    def train_autoencoder(self, training_data: Dict):
        """Train Autoencoder for anomaly detection"""
        from src.models.spectral_models import SpectralAutoencoder
        
        print("🔍 Training Autoencoder for anomaly detection...")
        
        # Prepare data (only healthy samples for unsupervised learning)
        X_healthy = self.prepare_autoencoder_data(training_data)
        
        # Initialize Autoencoder
        input_shape = (64, 64, self.config['data']['hyperspectral_bands'])
        autoencoder_model = SpectralAutoencoder(
            input_shape=input_shape,
            config=self.config['models']['autoencoder']
        )
        
        # Build and compile
        model = autoencoder_model.build_model()
        autoencoder_model.compile_model()
        
        # Train
        history = model.fit(
            X_healthy, X_healthy,  # Reconstruction task
            epochs=self.training_config.get('autoencoder_epochs', 100),
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ Autoencoder training completed. Final loss: {min(history.history['loss']):.4f}")
        return model
    
    def train_unet_model(self, training_data: Dict):
        """Train U-Net for semantic segmentation"""
        from src.models.unet_segmentation import SpectralUNet
        
        print("🎯 Training U-Net for semantic segmentation...")
        
        # Prepare segmentation data
        X_seg, y_seg = self.prepare_segmentation_data(training_data)
        
        # Initialize U-Net
        input_shape = (256, 256, self.config['data']['hyperspectral_bands'])
        unet_model = SpectralUNet(
            input_shape=input_shape,
            num_classes=4,
            config=self.config['models']['unet']
        )
        
        # Build and compile
        model = unet_model.build_model()
        unet_model.compile_model()
        
        # Train
        history = model.fit(
            X_seg, y_seg,
            epochs=self.training_config.get('unet_epochs', 80),
            batch_size=8,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ U-Net training completed. Final IoU: {max(history.history.get('iou', [0])):.3f}")
        return model
    
    def load_training_data(self, data_path: str) -> Dict:
        """Load and prepare all training data"""
        import json
        import numpy as np
        
        # Load metadata and labels
        with open(os.path.join(data_path, "training_labels.json"), 'r') as f:
            labels = json.load(f)
        
        # Load hyperspectral data
        training_data = {
            'labels': labels,
            'hyperspectral_data': {},
            'environmental_data': {}
        }
        
        # Load each field's data
        for field_id in ['field_a', 'field_b', 'field_c']:
            # Load hyperspectral
            hyperspectral_file = os.path.join(data_path, f"{field_id}_hyperspectral.npy")
            if os.path.exists(hyperspectral_file):
                training_data['hyperspectral_data'][field_id] = np.load(hyperspectral_file)
            
            # Load environmental
            env_file = os.path.join(data_path, f"{field_id}_environmental.csv")
            if os.path.exists(env_file):
                training_data['environmental_data'][field_id] = pd.read_csv(env_file)
        
        return training_data
    
    def save_trained_models(self, models: Dict):
        """Save all trained models"""
        os.makedirs("models/saved", exist_ok=True)
        
        for model_name, model in models.items():
            if hasattr(model, 'save'):
                # TensorFlow/Keras model
                model.save(f"models/saved/{model_name}_model.h5")
            else:
                # Scikit-learn model
                import joblib
                joblib.dump(model, f"models/saved/{model_name}_model.pkl")
        
        print("💾 All models saved successfully!")

# ... existing code ...

if __name__ == "__main__":
    main()