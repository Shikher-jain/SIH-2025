import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict

class SpectralUNet:
    """U-Net for pixel-level health mapping and segmentation"""
    
    def __init__(self, input_shape: Tuple, num_classes: int, config: Dict):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config
        self.model = None
        
    def conv_block(self, inputs, filters, dropout_rate=0.1):
        """Convolutional block with batch normalization and dropout"""
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x
    
    def encoder_block(self, inputs, filters, dropout_rate=0.1):
        """Encoder block with convolution and max pooling"""
        conv = self.conv_block(inputs, filters, dropout_rate)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool
    
    def decoder_block(self, inputs, skip_features, filters, dropout_rate=0.1):
        """Decoder block with upsampling and skip connections"""
        upsample = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        concat = layers.Concatenate()([upsample, skip_features])
        conv = self.conv_block(concat, filters, dropout_rate)
        return conv
    
    def build_model(self):
        """Build U-Net architecture for spectral segmentation"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        skip1, pool1 = self.encoder_block(inputs, 64)
        skip2, pool2 = self.encoder_block(pool1, 128)
        skip3, pool3 = self.encoder_block(pool2, 256)
        skip4, pool4 = self.encoder_block(pool3, 512)
        
        # Bottleneck
        bottleneck = self.conv_block(pool4, 1024)
        
        # Decoder
        decode4 = self.decoder_block(bottleneck, skip4, 512)
        decode3 = self.decoder_block(decode4, skip3, 256)
        decode2 = self.decoder_block(decode3, skip2, 128)
        decode1 = self.decoder_block(decode2, skip1, 64)
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(decode1)
        else:
            outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(decode1)
        
        self.model = keras.Model(inputs, outputs, name='SpectralUNet')
        return self.model
    
    def compile_model(self):
        """Compile U-Net model"""
        if self.num_classes == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
            
        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy', self.dice_coefficient, self.iou_score]
        )
    
    def dice_coefficient(self, y_true, y_pred):
        """Dice coefficient for segmentation evaluation"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + 
                                             tf.keras.backend.sum(y_pred_f) + smooth)
    
    def iou_score(self, y_true, y_pred):
        """Intersection over Union score"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def predict_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Predict segmentation mask for given image data"""
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
        
        prediction = self.model.predict(image_data)
        
        if self.num_classes == 1:
            # Binary segmentation
            return (prediction > 0.5).astype(np.uint8)
        else:
            # Multi-class segmentation
            return np.argmax(prediction, axis=-1)
    
    def generate_health_zones(self, segmentation_mask: np.ndarray) -> Dict:
        """Generate health zone statistics from segmentation mask"""
        zones = {}
        
        if self.num_classes == 1:
            # Binary: healthy (0) vs unhealthy (1)
            healthy_pixels = np.sum(segmentation_mask == 0)
            unhealthy_pixels = np.sum(segmentation_mask == 1)
            total_pixels = segmentation_mask.size
            
            zones = {
                'healthy_percentage': (healthy_pixels / total_pixels) * 100,
                'unhealthy_percentage': (unhealthy_pixels / total_pixels) * 100,
                'total_pixels': total_pixels,
                'unhealthy_areas': self._find_connected_components(segmentation_mask)
            }
        else:
            # Multi-class: healthy, stressed, diseased, etc.
            class_names = ['healthy', 'stressed', 'diseased', 'pest_damage']
            total_pixels = segmentation_mask.size
            
            for i, class_name in enumerate(class_names[:self.num_classes]):
                class_pixels = np.sum(segmentation_mask == i)
                zones[f'{class_name}_percentage'] = (class_pixels / total_pixels) * 100
                zones[f'{class_name}_areas'] = self._find_connected_components(
                    (segmentation_mask == i).astype(np.uint8)
                )
            
            zones['total_pixels'] = total_pixels
        
        return zones
    
    def _find_connected_components(self, binary_mask: np.ndarray) -> list:
        """Find connected components in binary mask"""
        from scipy import ndimage
        
        labeled_array, num_features = ndimage.label(binary_mask)
        components = []
        
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            component_coords = np.where(component_mask)
            
            # Calculate component properties
            area = np.sum(component_mask)
            centroid = (np.mean(component_coords[0]), np.mean(component_coords[1]))
            bbox = (
                np.min(component_coords[0]), np.min(component_coords[1]),
                np.max(component_coords[0]), np.max(component_coords[1])
            )
            
            components.append({
                'id': i,
                'area': area,
                'centroid': centroid,
                'bbox': bbox,
                'coordinates': list(zip(component_coords[0], component_coords[1]))
            })
        
        # Sort by area (largest first)
        components.sort(key=lambda x: x['area'], reverse=True)
        return components

class AttentionUNet(SpectralUNet):
    """Attention U-Net for improved segmentation with attention mechanisms"""
    
    def __init__(self, input_shape: Tuple, num_classes: int, config: Dict):
        super().__init__(input_shape, num_classes, config)
    
    def attention_block(self, g, x, inter_shape):
        """Attention block for focusing on relevant features"""
        # Gating signal
        theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
        
        # Input feature maps
        phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(g)
        
        # Concatenate and apply activation
        concat_xg = layers.add([theta_x, phi_g])
        act_xg = layers.Activation('relu')(concat_xg)
        
        # Apply convolution and sigmoid activation
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        
        # Upsample attention coefficients
        upsample_psi = layers.UpSampling2D(size=(2, 2))(sigmoid_xg)
        
        # Apply attention
        y = layers.multiply([upsample_psi, x])
        
        # Final convolution
        result = layers.Conv2D(inter_shape, (1, 1), padding='same')(y)
        result = layers.BatchNormalization()(result)
        
        return result
    
    def build_model(self):
        """Build Attention U-Net architecture"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        skip1, pool1 = self.encoder_block(inputs, 64)
        skip2, pool2 = self.encoder_block(pool1, 128)
        skip3, pool3 = self.encoder_block(pool2, 256)
        skip4, pool4 = self.encoder_block(pool3, 512)
        
        # Bottleneck
        bottleneck = self.conv_block(pool4, 1024)
        
        # Decoder with attention
        gating4 = layers.Conv2D(512, (1, 1), padding='same')(bottleneck)
        att4 = self.attention_block(gating4, skip4, 256)
        decode4 = self.decoder_block(bottleneck, att4, 512)
        
        gating3 = layers.Conv2D(256, (1, 1), padding='same')(decode4)
        att3 = self.attention_block(gating3, skip3, 128)
        decode3 = self.decoder_block(decode4, att3, 256)
        
        gating2 = layers.Conv2D(128, (1, 1), padding='same')(decode3)
        att2 = self.attention_block(gating2, skip2, 64)
        decode2 = self.decoder_block(decode3, att2, 128)
        
        gating1 = layers.Conv2D(64, (1, 1), padding='same')(decode2)
        att1 = self.attention_block(gating1, skip1, 32)
        decode1 = self.decoder_block(decode2, att1, 64)
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(decode1)
        else:
            outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(decode1)
        
        self.model = keras.Model(inputs, outputs, name='AttentionUNet')
        return self.model