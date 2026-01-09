"""
Data conversion and training script for spectral health mapping
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from datetime import datetime

from app.utils.preprocessing import HyperspectralProcessor, SensorDataProcessor
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataConverter:
    """Convert and prepare data for model training"""
    
    def __init__(self, data_dir: str = "data/sample"):
        """
        Initialize data converter
        
        Args:
            data_dir: Directory containing sample data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sample_data(self, num_samples: int = 100):
        """
        Create sample hyperspectral and sensor data for training
        
        Args:
            num_samples: Number of sample data points to create
        """
        logger.info(f"Creating {num_samples} sample data points...")
        
        # Create sample hyperspectral images
        images = []
        labels = []
        sensor_data = []
        
        health_categories = ['Critical', 'Poor', 'Fair', 'Good', 'Excellent']
        
        for i in range(num_samples):
            # Generate random health category
            health_idx = np.random.randint(0, len(health_categories))
            health_category = health_categories[health_idx]
            
            # Generate image based on health category
            if health_category == 'Critical':
                # Red/brown dominant
                image = self._generate_health_image([0.8, 0.3, 0.2], noise_level=0.3)
            elif health_category == 'Poor':
                # Yellow/orange dominant
                image = self._generate_health_image([0.9, 0.6, 0.3], noise_level=0.25)
            elif health_category == 'Fair':
                # Light green
                image = self._generate_health_image([0.6, 0.8, 0.4], noise_level=0.2)
            elif health_category == 'Good':
                # Green
                image = self._generate_health_image([0.4, 0.9, 0.3], noise_level=0.15)
            else:  # Excellent
                # Dark green
                image = self._generate_health_image([0.2, 0.9, 0.2], noise_level=0.1)
            
            images.append(image)
            labels.append(health_idx)
            
            # Generate corresponding sensor data
            sensor_values = self._generate_sensor_data(health_category)
            sensor_data.append(sensor_values)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        sensor_data = np.array(sensor_data)
        
        # Save data
        np.save(self.data_dir / "sample_images.npy", images)
        np.save(self.data_dir / "sample_labels.npy", labels)
        np.save(self.data_dir / "sample_sensor_data.npy", sensor_data)
        
        # Save metadata
        metadata = {
            'num_samples': num_samples,
            'health_categories': health_categories,
            'image_shape': images.shape[1:],
            'sensor_features': 9,
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Sample data created and saved to {self.data_dir}")
        return images, labels, sensor_data
    
    def _generate_health_image(self, base_color: list, noise_level: float = 0.2) -> np.ndarray:
        """Generate a health-based image"""
        # Create base image with health-related color
        image = np.random.normal(base_color, noise_level, (224, 224, 3))
        
        # Add some texture patterns
        x, y = np.meshgrid(np.linspace(0, 10, 224), np.linspace(0, 10, 224))
        pattern = 0.1 * np.sin(x) * np.cos(y)
        
        for c in range(3):
            image[:, :, c] += pattern
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        return image.astype(np.float32)
    
    def _generate_sensor_data(self, health_category: str) -> np.ndarray:
        """Generate sensor data correlated with health category"""
        if health_category == 'Critical':
            # Poor conditions
            data = {
                'air_temperature': np.random.normal(35, 5),      # High temp
                'humidity': np.random.normal(30, 10),            # Low humidity
                'soil_moisture': np.random.normal(20, 5),        # Low moisture
                'wind_speed': np.random.normal(8, 2),            # High wind
                'rainfall': np.random.normal(0.5, 0.5),          # Low rain
                'solar_radiation': np.random.normal(1200, 200),  # High radiation
                'leaf_wetness': np.random.normal(10, 5),         # Low wetness
                'co2_level': np.random.normal(450, 30),          # High CO2
                'ph_level': np.random.normal(8.5, 0.5)           # High pH
            }
        elif health_category == 'Poor':
            data = {
                'air_temperature': np.random.normal(30, 4),
                'humidity': np.random.normal(40, 8),
                'soil_moisture': np.random.normal(30, 5),
                'wind_speed': np.random.normal(6, 2),
                'rainfall': np.random.normal(1.5, 1),
                'solar_radiation': np.random.normal(1000, 150),
                'leaf_wetness': np.random.normal(20, 5),
                'co2_level': np.random.normal(430, 25),
                'ph_level': np.random.normal(8, 0.4)
            }
        elif health_category == 'Fair':
            data = {
                'air_temperature': np.random.normal(25, 3),
                'humidity': np.random.normal(55, 8),
                'soil_moisture': np.random.normal(45, 5),
                'wind_speed': np.random.normal(4, 1.5),
                'rainfall': np.random.normal(3, 1.5),
                'solar_radiation': np.random.normal(800, 100),
                'leaf_wetness': np.random.normal(35, 5),
                'co2_level': np.random.normal(410, 20),
                'ph_level': np.random.normal(7, 0.3)
            }
        elif health_category == 'Good':
            data = {
                'air_temperature': np.random.normal(22, 2),
                'humidity': np.random.normal(65, 5),
                'soil_moisture': np.random.normal(60, 5),
                'wind_speed': np.random.normal(3, 1),
                'rainfall': np.random.normal(5, 2),
                'solar_radiation': np.random.normal(600, 80),
                'leaf_wetness': np.random.normal(50, 5),
                'co2_level': np.random.normal(400, 15),
                'ph_level': np.random.normal(6.5, 0.2)
            }
        else:  # Excellent
            data = {
                'air_temperature': np.random.normal(20, 2),
                'humidity': np.random.normal(70, 5),
                'soil_moisture': np.random.normal(70, 3),
                'wind_speed': np.random.normal(2, 0.5),
                'rainfall': np.random.normal(7, 2),
                'solar_radiation': np.random.normal(500, 60),
                'leaf_wetness': np.random.normal(60, 5),
                'co2_level': np.random.normal(390, 10),
                'ph_level': np.random.normal(6.2, 0.15)
            }
        
        # Convert to array and normalize
        values = np.array([
            data['air_temperature'],
            data['humidity'],
            data['soil_moisture'],
            data['wind_speed'],
            data['rainfall'],
            data['solar_radiation'],
            data['leaf_wetness'],
            data['co2_level'],
            data['ph_level']
        ])
        
        # Apply realistic bounds
        values[0] = np.clip(values[0], -10, 50)     # temperature
        values[1] = np.clip(values[1], 0, 100)      # humidity
        values[2] = np.clip(values[2], 0, 100)      # soil moisture
        values[3] = np.clip(values[3], 0, 50)       # wind speed
        values[4] = np.clip(values[4], 0, 200)      # rainfall
        values[5] = np.clip(values[5], 0, 1500)     # solar radiation
        values[6] = np.clip(values[6], 0, 100)      # leaf wetness
        values[7] = np.clip(values[7], 300, 2000)   # CO2
        values[8] = np.clip(values[8], 0, 14)       # pH
        
        return values.astype(np.float32)


class ModelTrainer:
    """Train the CNN model for crop health prediction"""
    
    def __init__(self, data_dir: str = "data/sample"):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing training data
        """
        self.data_dir = Path(data_dir)
        self.model = None
        
    def load_data(self):
        """Load training data"""
        logger.info("Loading training data...")
        
        images = np.load(self.data_dir / "sample_images.npy")
        labels = np.load(self.data_dir / "sample_labels.npy")
        sensor_data = np.load(self.data_dir / "sample_sensor_data.npy")
        
        # Convert labels to categorical
        labels_categorical = tf.keras.utils.to_categorical(labels, num_classes=5)
        
        logger.info(f"Loaded {len(images)} samples")
        logger.info(f"Image shape: {images.shape}")
        logger.info(f"Sensor data shape: {sensor_data.shape}")
        
        return images, labels_categorical, sensor_data
    
    def create_model(self, input_shape: tuple = (224, 224, 3), num_classes: int = 5):
        """Create CNN model architecture"""
        logger.info("Creating CNN model...")
        
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
            
            # Global Average Pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu', name='features'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax', name='health_class')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Model created successfully")
        logger.info(f"Model parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, images, labels, validation_split=0.2, epochs=10):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, random_state=42
        )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
    
    def save_model(self, model_path: str = None):
        """Save the trained model"""
        if model_path is None:
            model_path = settings.MODEL_PATH
        
        # Create models directory
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def evaluate_model(self, images, labels):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        loss, accuracy = self.model.evaluate(images, labels, verbose=0)
        
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Model Loss: {loss:.4f}")
        
        return accuracy, loss


def main():
    """Main function for data conversion and model training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert data and train model")
    parser.add_argument("--create-data", action="store_true", help="Create sample data")
    parser.add_argument("--train-model", action="store_true", help="Train the model")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to create")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Initialize components
    data_converter = DataConverter()
    model_trainer = ModelTrainer()
    
    if args.create_data:
        logger.info("Creating sample data...")
        images, labels, sensor_data = data_converter.create_sample_data(args.samples)
        logger.info("Sample data creation completed")
    
    if args.train_model:
        logger.info("Starting model training pipeline...")
        
        # Load data
        images, labels, sensor_data = model_trainer.load_data()
        
        # Create and train model
        model = model_trainer.create_model()
        history = model_trainer.train_model(images, labels, epochs=args.epochs)
        
        # Evaluate model
        accuracy, loss = model_trainer.evaluate_model(images, labels)
        
        # Save model
        model_trainer.save_model()
        
        logger.info("Model training pipeline completed!")
        logger.info(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()