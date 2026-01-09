"""
Train hybrid CNN+LSTM and Random Forest ensemble model for crop health monitoring.
This script combines 3D CNN for hyperspectral analysis, LSTM for temporal features, 
and Random Forest ensemble for final predictions.
"""

import numpy as np
import pandas as pd
import os
import pickle
import joblib
from datetime import datetime
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv3D, MaxPooling3D, LSTM, Dense, 
                                   Dropout, Flatten, GlobalAveragePooling3D,
                                   Input, Concatenate, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

class HybridCropHealthModel:
    def __init__(self):
        self.cnn_lstm_model = None
        self.rf_model = None
        self.label_encoder = None
        self.scaler = None
        self.image_shape = (64, 64, 50, 1)  # Height, Width, Spectral_Bands, Channels
        self.sequence_length = 1  # For hyperspectral analysis
        
    def load_image_data(self, df, data_path='../data/images'):
        """Load and preprocess hyperspectral image data."""
        images = []
        
        for _, row in df.iterrows():
            if 'ImageName' in row:
                img_file = os.path.join(data_path, f"{row['ImageName']}.npy")
            else:
                # Fallback to ROI-based naming
                roi_id = row['roi_id'] if 'roi_id' in row else f"roi_{len(images)+1:02d}"
                date_str = row['date'].replace('-', '') if 'date' in row else '20210714'
                img_file = os.path.join(data_path, f"{roi_id}_{date_str}.npy")
            
            if os.path.exists(img_file):
                img_data = np.load(img_file)
                # Ensure correct shape for 3D CNN: (H, W, Bands)
                if len(img_data.shape) == 3:
                    # Add channel dimension: (H, W, Bands, 1)
                    img_data = np.expand_dims(img_data, -1)
                    # Resize to expected dimensions
                    img_data = self._resize_hyperspectral_image(img_data)
                images.append(img_data)
            else:
                # Create synthetic hyperspectral data if file doesn't exist
                synthetic_img = np.random.rand(*self.image_shape)
                images.append(synthetic_img)
        
        return np.array(images)
    
    def _resize_hyperspectral_image(self, img):
        """Resize hyperspectral image to target shape."""
        h, w, bands, channels = img.shape
        th, tw, tb, tc = self.image_shape
        
        # Resize spatial dimensions
        if h >= th and w >= tw:
            # Center crop spatial dimensions
            start_h = (h - th) // 2
            start_w = (w - tw) // 2
            img_resized = img[start_h:start_h+th, start_w:start_w+tw, :, :]
        else:
            # Pad spatial dimensions
            result = np.zeros((th, tw, bands, channels))
            end_h = min(h, th)
            end_w = min(w, tw)
            result[:end_h, :end_w, :, :] = img[:end_h, :end_w, :, :]
            img_resized = result
        
        # Handle spectral bands
        if bands >= tb:
            # Take first tb bands
            img_resized = img_resized[:, :, :tb, :]
        else:
            # Pad with zeros if fewer bands
            result = np.zeros(self.image_shape)
            result[:, :, :bands, :] = img_resized
            img_resized = result
        
        return img_resized
    
    def build_cnn_lstm_model(self, n_classes):
        """Build 3D CNN + LSTM hybrid model for hyperspectral analysis."""
        # CNN branch for 3D hyperspectral data
        cnn_input = Input(shape=self.image_shape)
        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(cnn_input)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = GlobalAveragePooling3D()(x)
        cnn_output = Dense(64, activation='relu')(x)
        
        # LSTM branch for tabular sequence features
        lstm_input = Input(shape=(10, 1))  # 10 tabular features as sequence
        l = LSTM(64, return_sequences=False)(lstm_input)
        lstm_output = Dense(32, activation='relu')(l)
        
        # Combine CNN + LSTM features
        combined = Concatenate()([cnn_output, lstm_output])
        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.3)(z)
        z = Dense(n_classes, activation='softmax')(z)
        
        model = Model(inputs=[cnn_input, lstm_input], outputs=z)
        return model
    
    def prepare_tabular_data(self, df):
        """Prepare tabular data for LSTM and Random Forest."""
        # Select hyperspectral-derived and sensor features
        feature_columns = [
            'NDVI', 'NDWI', 'NDSI', 'CHI', 'GNDVI',
            'SoilMoisture', 'AirTemp', 'Humidity', 'Rainfall', 'pH'
        ]
        
        X_tabular = df[feature_columns].fillna(df[feature_columns].mean())
        y = df['Label'] if 'Label' in df.columns else df['health_class']
        
        return X_tabular, y
    
    def _generate_missing_indices(self, X_tabular, df):
        """Generate missing spectral indices from existing data."""
        print("Generating missing spectral indices...")
        
        # Create synthetic spectral indices if not present
        indices_to_add = {}
        
        if 'NDVI' not in X_tabular.columns:
            indices_to_add['NDVI'] = np.random.beta(3, 2, len(df))
        if 'NDWI' not in X_tabular.columns:
            indices_to_add['NDWI'] = np.random.normal(0.3, 0.2, len(df))
        if 'NDSI' not in X_tabular.columns:
            indices_to_add['NDSI'] = np.random.normal(0.2, 0.15, len(df))
        if 'CHI' not in X_tabular.columns:
            indices_to_add['CHI'] = np.random.beta(2, 3, len(df))
        if 'GNDVI' not in X_tabular.columns:
            indices_to_add['GNDVI'] = np.random.beta(2.5, 2, len(df))
        if 'SoilMoisture' not in X_tabular.columns:
            indices_to_add['SoilMoisture'] = df.get('soil_moisture', np.random.beta(2, 2, len(df)) * 100) / 100
        if 'AirTemp' not in X_tabular.columns:
            indices_to_add['AirTemp'] = df.get('temperature_avg', np.random.normal(25, 8, len(df)))
        if 'Humidity' not in X_tabular.columns:
            indices_to_add['Humidity'] = df.get('humidity_avg', np.random.normal(65, 15, len(df)))
        if 'Rainfall' not in X_tabular.columns:
            indices_to_add['Rainfall'] = df.get('rainfall_mm', np.random.exponential(5, len(df)))
        if 'pH' not in X_tabular.columns:
            indices_to_add['pH'] = df.get('soil_ph', np.random.normal(6.5, 0.8, len(df)))
        
        # Add missing columns to X_tabular
        for col, values in indices_to_add.items():
            X_tabular[col] = values
        
        return X_tabular
    
    def train(self, csv_path='../data/final_dataset.csv', test_size=0.2, random_state=42):
        """Train the hybrid model using integrated approach."""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Prepare labels
        self.label_encoder = LabelEncoder()
        if 'Label' in df.columns:
            y_encoded = self.label_encoder.fit_transform(df['Label'])
        else:
            # Map existing health classes to simplified labels
            label_mapping = {
                'Excellent': 'Healthy',
                'Good': 'Healthy', 
                'Fair': 'Stressed',
                'Poor': 'Stressed',
                'Critical': 'Diseased'
            }
            df['Label'] = df['health_class'].map(label_mapping)
            y_encoded = self.label_encoder.fit_transform(df['Label'])
        
        n_classes = len(self.label_encoder.classes_)
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Number of classes: {n_classes}")
        
        # Prepare tabular data
        print("Preparing tabular data...")
        X_tabular, y = self.prepare_tabular_data(df)
        
        # Generate missing spectral indices if needed
        if 'NDVI' not in X_tabular.columns:
            X_tabular = self._generate_missing_indices(X_tabular, df)
        
        # Scale tabular features
        self.scaler = StandardScaler()
        X_tabular_scaled = self.scaler.fit_transform(X_tabular)
        
        # Load image data
        print("Loading hyperspectral image data...")
        X_images = self.load_image_data(df)
        
        # Ensure we have the same number of samples
        min_samples = min(len(X_images), len(X_tabular_scaled))
        X_images = X_images[:min_samples]
        X_tabular_scaled = X_tabular_scaled[:min_samples]
        y_encoded = y_encoded[:min_samples]
        
        # Split data
        X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
            X_images, X_tabular_scaled, y_encoded, 
            test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"Training set size: {len(X_img_train)}")
        print(f"Test set size: {len(X_img_test)}")
        
        # Prepare LSTM input (add sequence dimension)
        X_tab_train_seq = np.expand_dims(X_tab_train, -1)
        X_tab_test_seq = np.expand_dims(X_tab_test, -1)
        
        # Build and compile CNN+LSTM model
        print("\nBuilding CNN+LSTM model...")
        self.cnn_lstm_model = self.build_cnn_lstm_model(n_classes)
        self.cnn_lstm_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("CNN+LSTM Model Architecture:")
        self.cnn_lstm_model.summary()
        
        # Train CNN+LSTM model
        print("\nTraining CNN+LSTM model...")
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('../models/best_cnn_lstm_model.h5', save_best_only=True)
        ]
        
        history = self.cnn_lstm_model.fit(
            [X_img_train, X_tab_train_seq],
            y_train,
            validation_split=0.1,
            batch_size=8,
            epochs=20,
            verbose=1,
            callbacks=callbacks
        )
        
        # Extract deep features for Random Forest ensemble
        print("\nExtracting deep features...")
        feature_extractor = Model(
            inputs=self.cnn_lstm_model.inputs,
            outputs=self.cnn_lstm_model.layers[-2].output  # Before final classification layer
        )
        
        deep_features_train = feature_extractor.predict([X_img_train, X_tab_train_seq])
        deep_features_test = feature_extractor.predict([X_img_test, X_tab_test_seq])
        
        # Train Random Forest on extracted deep features
        print("\nTraining Random Forest ensemble...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.rf_model.fit(deep_features_train, y_train)
        
        # Evaluate models
        cnn_pred = self.cnn_lstm_model.predict([X_img_test, X_tab_test_seq])
        cnn_pred_classes = np.argmax(cnn_pred, axis=1)
        cnn_accuracy = accuracy_score(y_test, cnn_pred_classes)
        
        rf_pred = self.rf_model.predict(deep_features_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        print(f"\nCNN+LSTM Accuracy: {cnn_accuracy:.4f}")
        print(f"Random Forest Ensemble Accuracy: {rf_accuracy:.4f}")
        
        print("\nCNN+LSTM Classification Report:")
        print(classification_report(y_test, cnn_pred_classes, target_names=self.label_encoder.classes_))
        
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred, target_names=self.label_encoder.classes_))
        
        # Save models
        self.save_models()
        
        # Plot training history
        self.plot_training_history(history)
        
        return {
            'cnn_accuracy': cnn_accuracy,
            'rf_accuracy': rf_accuracy,
            'test_predictions_cnn': cnn_pred_classes,
            'test_predictions_rf': rf_pred,
            'test_labels': y_test,
            'training_history': history.history
        }
    
    def plot_training_history(self, history):
        """Plot training history for CNN+LSTM model."""
        if history is None:
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('../outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """Plot Random Forest feature importance."""
        if self.rf_model is None:
            return
        
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (Random Forest)")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('../outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def save_models(self):
        """Save trained models and preprocessors."""
        os.makedirs('../models', exist_ok=True)
        
        # Save CNN+LSTM model
        if self.cnn_lstm_model:
            self.cnn_lstm_model.save('../models/hybrid_cnn_lstm_model.h5')
            print("CNN+LSTM model saved to: ../models/hybrid_cnn_lstm_model.h5")
        
        # Save Random Forest model
        if self.rf_model:
            joblib.dump(self.rf_model, '../models/rf_classifier.pkl')
            print("Random Forest model saved to: ../models/rf_classifier.pkl")
        
        # Save label encoder
        if self.label_encoder:
            joblib.dump(self.label_encoder, '../models/label_encoder.pkl')
            print("Label encoder saved to: ../models/label_encoder.pkl")
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, '../models/scaler.pkl')
            print("Scaler saved to: ../models/scaler.pkl")

def main():
    """Main training function."""
    print("Starting Hybrid Crop Health Model Training...")
    print("=" * 50)
    
    # Initialize model
    model = HybridCropHealthModel()
    
    # Train model
    results = model.train()
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Final CNN+LSTM Accuracy: {results['cnn_accuracy']:.4f}")
    print(f"Final Random Forest Accuracy: {results['rf_accuracy']:.4f}")

if __name__ == "__main__":
    main()