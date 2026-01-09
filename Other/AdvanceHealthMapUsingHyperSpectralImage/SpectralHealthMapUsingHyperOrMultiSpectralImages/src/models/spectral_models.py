import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict

class SpectralCNN:
    """CNN for spectral data classification and feature extraction"""
    
    def __init__(self, input_shape: Tuple, num_classes: int, config: Dict):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config
        self.model = None
        
    def build_model(self) -> keras.Model:
        """Build CNN architecture for spectral analysis"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial spectral convolution
        x = layers.Conv3D(32, (3, 3, 7), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 1, 2))(x)
        
        # Spatial-spectral feature extraction
        x = layers.Conv3D(64, (3, 3, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling3D()(x)
        
        # Classification head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs, name='SpectralCNN')
        return self.model
    
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, train_data, train_labels, validation_data, epochs=100):
        """Train the CNN model"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint('models/saved/best_spectral_cnn.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks
        )
        
        return history

class SpectralAutoencoder:
    """Autoencoder for anomaly detection in spectral data"""
    
    def __init__(self, input_shape: Tuple, config: Dict):
        self.input_shape = input_shape
        self.config = config
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        
    def build_model(self):
        """Build autoencoder architecture"""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Bottleneck
        encoded = layers.Flatten()(x)
        encoded = layers.Dense(self.config['encoding_dim'], activation='relu')(encoded)
        
        self.encoder = keras.Model(encoder_input, encoded, name='encoder')
        
        # Decoder
        decoder_input = keras.Input(shape=(self.config['encoding_dim'],))
        x = layers.Dense(np.prod(self.input_shape) // 64, activation='relu')(decoder_input)
        x = layers.Reshape((self.input_shape[0]//8, self.input_shape[1]//8, 256))(x)
        
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        decoded = layers.Conv2D(self.input_shape[-1], (3, 3), 
                               activation='sigmoid', padding='same')(x)
        
        self.decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Autoencoder
        autoencoder_input = keras.Input(shape=self.input_shape)
        encoded_img = self.encoder(autoencoder_input)
        decoded_img = self.decoder(encoded_img)
        
        self.autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
        
        return self.autoencoder
    
    def compile_model(self):
        """Compile autoencoder"""
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    def detect_anomalies(self, data: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using reconstruction error"""
        reconstructed = self.autoencoder.predict(data)
        reconstruction_error = np.mean(np.square(data - reconstructed), axis=-1)
        
        if threshold is None:
            threshold = np.percentile(reconstruction_error, 95)
        
        anomalies = reconstruction_error > threshold
        return anomalies, reconstruction_error

class SpectralLSTM:
    """LSTM for temporal analysis of spectral data"""
    
    def __init__(self, input_shape: Tuple, config: Dict):
        self.input_shape = input_shape
        self.config = config
        self.model = None
        
    def build_model(self):
        """Build LSTM architecture for temporal spectral analysis"""
        inputs = keras.Input(shape=self.input_shape)
        
        # LSTM layers
        x = layers.LSTM(self.config['units'], return_sequences=True, 
                       dropout=self.config['dropout'])(inputs)
        x = layers.LSTM(self.config['units'] // 2, return_sequences=True,
                       dropout=self.config['dropout'])(x)
        x = layers.LSTM(self.config['units'] // 4, dropout=self.config['dropout'])(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output for prediction
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Disease probability
        
        self.model = keras.Model(inputs, outputs, name='SpectralLSTM')
        return self.model
    
    def compile_model(self):
        """Compile LSTM model"""
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

class HybridCNNLSTM:
    """Hybrid CNN-LSTM model for spatio-temporal analysis"""
    
    def __init__(self, spatial_shape: Tuple, temporal_length: int, config: Dict):
        self.spatial_shape = spatial_shape
        self.temporal_length = temporal_length
        self.config = config
        self.model = None
        
    def build_model(self):
        """Build hybrid CNN-LSTM architecture"""
        # Input for time-distributed CNN
        inputs = keras.Input(shape=(self.temporal_length,) + self.spatial_shape)
        
        # Time-distributed CNN for spatial feature extraction
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(inputs)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # LSTM for temporal modeling
        x = layers.LSTM(256, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(128, dropout=0.2)(x)
        
        # Output heads
        disease_classification = layers.Dense(4, activation='softmax', name='disease_type')(x)
        severity_estimation = layers.Dense(1, activation='sigmoid', name='severity')(x)
        progression_prediction = layers.Dense(7, activation='sigmoid', name='progression')(x)  # 7-day forecast
        
        self.model = keras.Model(
            inputs=inputs,
            outputs=[disease_classification, severity_estimation, progression_prediction],
            name='HybridCNNLSTM'
        )
        
        return self.model
    
    def compile_model(self):
        """Compile hybrid model"""
        self.model.compile(
            optimizer='adam',
            loss={
                'disease_type': 'categorical_crossentropy',
                'severity': 'mse',
                'progression': 'mse'
            },
            loss_weights={
                'disease_type': 1.0,
                'severity': 0.5,
                'progression': 0.3
            },
            metrics={
                'disease_type': ['accuracy'],
                'severity': ['mae'],
                'progression': ['mae']
            }
        )