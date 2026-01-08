# Chapter 4: Data Processing and Analysis

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the complete data processing pipeline of the spectral health mapping system
- Explain the preprocessing steps applied to hyperspectral and environmental data
- Describe how vegetation indices are calculated and used
- Identify feature extraction techniques for spectral data
- Implement data processing workflows using Python

## Key Concepts

### Data Processing Pipeline Overview

The AI-Powered Spectral Health Mapping System follows a comprehensive data processing pipeline:

```
Raw Input Data
├── Hyperspectral Images (224 bands)
├── Environmental Sensor Data (CSV)
└── Field Metadata (JSON)

Preprocessing Stage
├── Spectral Preprocessing
│   ├── Atmospheric Correction
│   ├── Noise Reduction
│   └── Vegetation Index Calculation
├── Environmental Data Processing
│   ├── Data Validation
│   ├── Normalization
│   └── Feature Engineering
└── Data Integration
    ├── Spatial Alignment
    └── Temporal Synchronization

Analysis Stage
├── AI Model Processing
│   ├── Disease Detection (CNN)
│   ├── Anomaly Detection (Autoencoder)
│   ├── Temporal Analysis (LSTM)
│   └── Health Segmentation (U-Net)
├── Risk Assessment
│   ├── Environmental Risk
│   ├── Spectral Risk
│   └── Temporal Risk
└── Recommendation Generation
    ├── Treatment Planning
    └── Monitoring Scheduling

Output Generation
├── Health Maps
├── Risk Maps
├── Alerts
└── Recommendations
```

## Spectral Data Preprocessing

### Atmospheric Correction

Atmospheric conditions affect the spectral signatures captured by sensors. Atmospheric correction removes these effects:

```python
class SpectralProcessor:
    def atmospheric_correction(self, data):
        """
        Apply atmospheric correction using dark object subtraction
        """
        corrected = data.copy()
        for band in range(data.shape[0]):
            # Find dark value (1st percentile)
            dark_value = np.percentile(data[band], 1)
            # Subtract dark value
            corrected[band] = np.clip(data[band] - dark_value, 0, None)
        return corrected
```

### Noise Reduction

Hyperspectral data often contains various types of noise that need to be reduced:

```python
def noise_reduction(self, data, method='gaussian'):
    """
    Apply noise reduction techniques
    """
    if method == 'gaussian':
        return ndimage.gaussian_filter(data, sigma=1.0)
    elif method == 'median':
        return ndimage.median_filter(data, size=3)
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        return savgol_filter(data, window_length=5, polyorder=2, axis=0)
    return data
```

### Vegetation Index Calculation

Vegetation indices are crucial for crop health assessment:

```python
def calculate_vegetation_indices(self, data, wavelengths):
    """
    Calculate various vegetation indices
    """
    indices = {}
    
    # Find bands closest to required wavelengths
    red_idx = np.argmin(np.abs(wavelengths - 670))  # Red (~670nm)
    nir_idx = np.argmin(np.abs(wavelengths - 800))  # NIR (~800nm)
    green_idx = np.argmin(np.abs(wavelengths - 550))  # Green (~550nm)
    blue_idx = np.argmin(np.abs(wavelengths - 450))  # Blue (~450nm)
    
    red = data[red_idx]
    nir = data[nir_idx]
    green = data[green_idx]
    blue = data[blue_idx]
    
    # NDVI - Normalized Difference Vegetation Index
    indices['NDVI'] = (nir - red) / (nir + red + 1e-8)
    
    # EVI - Enhanced Vegetation Index
    indices['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    
    # SAVI - Soil Adjusted Vegetation Index
    L = 0.5
    indices['SAVI'] = ((nir - red) / (nir + red + L)) * (1 + L)
    
    # GNDVI - Green Normalized Difference Vegetation Index
    indices['GNDVI'] = (nir - green) / (nir + green + 1e-8)
    
    return indices
```

### Spectral Feature Extraction

Advanced spectral features provide more detailed information:

```python
def extract_spectral_features(self, data):
    """
    Extract advanced spectral features
    """
    features = []
    
    # Statistical features
    features.append(np.mean(data, axis=-1))  # Mean
    features.append(np.std(data, axis=-1))   # Standard deviation
    features.append(np.min(data, axis=-1))   # Minimum
    features.append(np.max(data, axis=-1))   # Maximum
    features.append(np.median(data, axis=-1)) # Median
    
    # Spectral derivatives
    derivatives = np.gradient(data, axis=-1)
    features.append(np.mean(derivatives, axis=-1))
    features.append(np.std(derivatives, axis=-1))
    
    # Red edge features
    red_edge_start = np.argmax(derivatives, axis=-1)
    features.append(red_edge_start)
    
    return np.stack(features, axis=-1)
```

## Environmental Data Processing

### Data Validation and Cleaning

Environmental sensor data requires validation and cleaning:

```python
class EnvironmentalDataProcessor:
    def validate_sensor_data(self, sensor_data):
        """
        Validate environmental sensor data
        """
        required_fields = [
            'air_temperature', 'soil_temperature', 'humidity',
            'soil_moisture', 'wind_speed', 'rainfall',
            'solar_radiation', 'leaf_wetness', 'co2_level', 'ph_level'
        ]
        
        # Check for missing fields
        for field in required_fields:
            if field not in sensor_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Check value ranges
        if not (sensor_data['air_temperature'] > -50 and sensor_data['air_temperature'] < 60):
            raise ValueError("Air temperature out of valid range")
        
        if not (sensor_data['humidity'] >= 0 and sensor_data['humidity'] <= 100):
            raise ValueError("Humidity out of valid range")
        
        return True
```

### Data Normalization

Normalizing environmental data for consistent processing:

```python
def process_sensor_data(self, sensor_data):
    """
    Process and normalize sensor data
    """
    processed_features = []
    features = [
        'soil_moisture', 'soil_temperature', 'air_temperature',
        'humidity', 'wind_speed', 'rainfall', 'solar_radiation',
        'leaf_wetness', 'co2_level', 'ph_level'
    ]
    
    for feature in features:
        if feature in sensor_data:
            # Normalize based on expected ranges
            if feature == 'soil_moisture':
                normalized = sensor_data[feature] / 100.0  # 0-100%
            elif feature in ['soil_temperature', 'air_temperature']:
                normalized = (sensor_data[feature] + 20) / 60.0  # -20 to 40°C
            elif feature == 'humidity':
                normalized = sensor_data[feature] / 100.0  # 0-100%
            elif feature == 'wind_speed':
                normalized = sensor_data[feature] / 30.0  # 0-30 m/s
            elif feature == 'rainfall':
                normalized = np.clip(sensor_data[feature] / 50.0, 0, 1)  # 0-50mm
            elif feature == 'solar_radiation':
                normalized = sensor_data[feature] / 1000.0  # 0-1000 W/m²
            elif feature == 'leaf_wetness':
                normalized = sensor_data[feature] / 100.0  # 0-100%
            elif feature == 'co2_level':
                normalized = (sensor_data[feature] - 300) / 500.0  # 300-800 ppm
            elif feature == 'ph_level':
                normalized = (sensor_data[feature] - 4) / 6.0  # 4-10 pH
            else:
                normalized = sensor_data[feature]
            
            processed_features.append(normalized)
        else:
            processed_features.append(0.0)  # Default value
    
    return np.array(processed_features)
```

## Data Integration

### Spatial Alignment

Ensuring hyperspectral and environmental data align spatially:

```python
def align_spatial_data(hyperspectral_data, environmental_data, field_metadata):
    """
    Align hyperspectral and environmental data spatially
    """
    # Get field dimensions
    height, width = hyperspectral_data.shape[1], hyperspectral_data.shape[2]
    
    # Interpolate environmental data to match field dimensions
    # This is a simplified example - real implementation would be more complex
    aligned_env_data = np.tile(environmental_data, (height, width, 1))
    
    return hyperspectral_data, aligned_env_data
```

### Temporal Synchronization

Aligning data across different time periods:

```python
def create_temporal_features(self, sensor_history, window_size=7):
    """
    Create temporal features from sensor history
    """
    temporal_data = []
    
    for i in range(len(sensor_history) - window_size + 1):
        window_data = []
        for j in range(i, i + window_size):
            features = self.process_sensor_data(sensor_history[j])
            window_data.append(features)
        temporal_data.append(window_data)
    
    return np.array(temporal_data)
```

## AI Model Processing

### Data Preparation for Models

Preparing data in the format required by different AI models:

```python
def prepare_data_for_models(self, processed_data):
    """
    Prepare data for different AI models
    """
    model_inputs = {}
    
    # For CNN - create patches
    model_inputs['cnn_input'] = self.create_data_patches(
        processed_data['hyperspectral'], patch_size=64, stride=32
    )
    
    # For Autoencoder - use full data
    model_inputs['autoencoder_input'] = processed_data['hyperspectral']
    
    # For LSTM - create time series
    model_inputs['lstm_input'] = self.create_temporal_series(
        processed_data['spectral_features']
    )
    
    # For U-Net - use full field data
    model_inputs['unet_input'] = processed_data['hyperspectral']
    
    return model_inputs
```

### Model Execution

Running the AI models on prepared data:

```python
def run_ai_models(self, model_inputs, models):
    """
    Run AI models on prepared inputs
    """
    results = {}
    
    # Run CNN for disease detection
    if 'cnn' in models and 'cnn_input' in model_inputs:
        cnn_results = models['cnn'].predict(model_inputs['cnn_input'])
        results['disease_detection'] = {
            'probabilities': cnn_results,
            'predicted_classes': np.argmax(cnn_results, axis=1)
        }
    
    # Run Autoencoder for anomaly detection
    if 'autoencoder' in models and 'autoencoder_input' in model_inputs:
        reconstructed = models['autoencoder'].predict(model_inputs['autoencoder_input'])
        reconstruction_error = np.mean(
            np.square(model_inputs['autoencoder_input'] - reconstructed), 
            axis=-1
        )
        results['anomaly_detection'] = {
            'reconstruction_errors': reconstruction_error,
            'anomaly_map': reconstruction_error > np.percentile(reconstruction_error, 95)
        }
    
    # Run LSTM for temporal analysis
    if 'lstm' in models and 'lstm_input' in model_inputs:
        lstm_results = models['lstm'].predict(model_inputs['lstm_input'])
        results['temporal_analysis'] = {
            'disease_progression': lstm_results
        }
    
    # Run U-Net for segmentation
    if 'unet' in models and 'unet_input' in model_inputs:
        segmentation = models['unet'].predict(
            np.expand_dims(model_inputs['unet_input'], axis=0)
        )
        results['health_segmentation'] = {
            'health_map': np.argmax(segmentation[0], axis=-1)
        }
    
    return results
```

## Risk Assessment

### Environmental Risk Calculation

```python
def calculate_environmental_risk(self, sensor_data):
    """
    Calculate environmental risk score
    """
    risk_score = 0.0
    
    # Temperature stress
    temp = sensor_data.get('air_temperature', 20)
    if temp > 35 or temp < 5:
        risk_score += 0.3
    elif temp > 30 or temp < 10:
        risk_score += 0.1
        
    # Humidity risk (disease favorable conditions)
    humidity = sensor_data.get('humidity', 50)
    if humidity > 85:
        risk_score += 0.25
    elif humidity > 70:
        risk_score += 0.1
        
    # Rainfall impact
    rainfall = sensor_data.get('rainfall', 0)
    if rainfall > 10:  # Heavy rain can spread diseases
        risk_score += 0.2
    elif rainfall < 1:  # Drought stress
        risk_score += 0.15
        
    # Leaf wetness duration
    leaf_wetness = sensor_data.get('leaf_wetness', 0)
    if leaf_wetness > 6:  # Hours of wetness
        risk_score += 0.25
        
    return min(risk_score, 1.0)
```

### Spectral Risk Analysis

```python
def calculate_spectral_risk(self, spectral_indices, anomaly_scores):
    """
    Calculate risk based on spectral analysis
    """
    risk_score = 0.0
    
    # NDVI decline
    ndvi = spectral_indices.get('NDVI', 0.8)
    if ndvi < 0.3:
        risk_score += 0.4
    elif ndvi < 0.5:
        risk_score += 0.2
        
    # Stress indicators
    if spectral_indices.get('PRI', 0) < -0.1:  # Photochemical stress
        risk_score += 0.2
        
    # Anomaly detection
    anomaly_percentage = np.mean(anomaly_scores > 0.7)
    risk_score += anomaly_percentage * 0.4
    
    return min(risk_score, 1.0)
```

## Practical Exercises

### Exercise 1: Implementing a Spectral Processor

```python
import numpy as np
from scipy import ndimage

class SimpleSpectralProcessor:
    def __init__(self):
        self.wavelengths = np.linspace(400, 2500, 224)  # 224 bands
    
    def process_hyperspectral_data(self, raw_data):
        """
        Complete preprocessing pipeline for hyperspectral data
        """
        print("Starting hyperspectral data processing...")
        
        # Step 1: Atmospheric correction
        corrected_data = self.atmospheric_correction(raw_data)
        print("Atmospheric correction completed")
        
        # Step 2: Noise reduction
        denoised_data = self.noise_reduction(corrected_data)
        print("Noise reduction completed")
        
        # Step 3: Calculate vegetation indices
        veg_indices = self.calculate_vegetation_indices(denoised_data)
        print("Vegetation indices calculated")
        
        # Step 4: Extract spectral features
        spectral_features = self.extract_spectral_features(denoised_data)
        print("Spectral features extracted")
        
        return {
            'corrected_data': corrected_data,
            'denoised_data': denoised_data,
            'vegetation_indices': veg_indices,
            'spectral_features': spectral_features
        }
    
    def atmospheric_correction(self, data):
        """Apply dark object subtraction"""
        corrected = data.copy()
        for band in range(data.shape[0]):
            dark_value = np.percentile(data[band], 1)
            corrected[band] = np.clip(data[band] - dark_value, 0, None)
        return corrected
    
    def noise_reduction(self, data):
        """Apply Gaussian filtering"""
        return ndimage.gaussian_filter(data, sigma=1.0)
    
    def calculate_vegetation_indices(self, data):
        """Calculate NDVI"""
        red_idx = np.argmin(np.abs(self.wavelengths - 670))
        nir_idx = np.argmin(np.abs(self.wavelengths - 800))
        
        red = data[red_idx]
        nir = data[nir_idx]
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        return {'NDVI': ndvi}
    
    def extract_spectral_features(self, data):
        """Extract basic statistical features"""
        features = np.stack([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            np.max(data, axis=0),
            np.min(data, axis=0)
        ], axis=-1)
        return features

# Example usage
processor = SimpleSpectralProcessor()

# Create sample hyperspectral data (224 bands, 64x64 pixels)
sample_data = np.random.random((224, 64, 64))

# Process the data
results = processor.process_hyperspectral_data(sample_data)
print(f"Processing complete. Results keys: {list(results.keys())}")
```

### Exercise 2: Environmental Data Processing

```python
class SimpleEnvironmentalProcessor:
    def process_environmental_data(self, sensor_data):
        """
        Process environmental sensor data
        """
        print("Processing environmental data...")
        
        # Validate data
        self.validate_sensor_data(sensor_data)
        
        # Normalize data
        normalized_data = self.normalize_sensor_data(sensor_data)
        
        # Calculate risk factors
        risk_score = self.calculate_environmental_risk(sensor_data)
        
        return {
            'normalized_data': normalized_data,
            'risk_score': risk_score
        }
    
    def validate_sensor_data(self, sensor_data):
        """Basic validation"""
        required_fields = ['air_temperature', 'humidity', 'soil_moisture']
        for field in required_fields:
            if field not in sensor_data:
                raise ValueError(f"Missing required field: {field}")
    
    def normalize_sensor_data(self, sensor_data):
        """Normalize sensor values to 0-1 range"""
        normalized = {}
        
        # Temperature normalization (-20 to 40°C)
        if 'air_temperature' in sensor_data:
            normalized['air_temperature'] = (sensor_data['air_temperature'] + 20) / 60
        
        # Humidity normalization (0-100%)
        if 'humidity' in sensor_data:
            normalized['humidity'] = sensor_data['humidity'] / 100
        
        # Soil moisture normalization (0-100%)
        if 'soil_moisture' in sensor_data:
            normalized['soil_moisture'] = sensor_data['soil_moisture'] / 100
            
        return normalized
    
    def calculate_environmental_risk(self, sensor_data):
        """Calculate simple environmental risk"""
        risk = 0.0
        
        # Temperature extremes
        if sensor_data.get('air_temperature', 20) > 35:
            risk += 0.3
        elif sensor_data.get('air_temperature', 20) < 5:
            risk += 0.3
            
        # High humidity
        if sensor_data.get('humidity', 50) > 80:
            risk += 0.2
            
        # Low soil moisture
        if sensor_data.get('soil_moisture', 50) < 20:
            risk += 0.25
            
        return min(risk, 1.0)

# Example usage
env_processor = SimpleEnvironmentalProcessor()

# Sample sensor data
sample_sensor_data = {
    'air_temperature': 28.5,
    'humidity': 75,
    'soil_moisture': 35,
    'wind_speed': 3.2,
    'rainfall': 0.0
}

# Process environmental data
env_results = env_processor.process_environmental_data(sample_sensor_data)
print(f"Environmental processing complete.")
print(f"Normalized data: {env_results['normalized_data']}")
print(f"Environmental risk score: {env_results['risk_score']:.2f}")
```

## Discussion Questions

1. Why is atmospheric correction important in hyperspectral data processing?
2. How do different noise reduction techniques affect the quality of vegetation indices?
3. What are the challenges in aligning hyperspectral and environmental data spatially and temporally?
4. How can we ensure that the preprocessing steps don't introduce artifacts into the data?

## Additional Resources

- Schaepman, M.E. "Radiation Physics and Spectral Analysis"
- Thenkabail, P.S. "Hyperspectral Remote Sensing of Vegetation"
- TensorFlow documentation on data preprocessing
- Scikit-learn preprocessing documentation

## Summary

This chapter covered the comprehensive data processing pipeline of the AI-Powered Spectral Health Mapping System. We explored preprocessing techniques for both hyperspectral and environmental data, including atmospheric correction, noise reduction, vegetation index calculation, and feature extraction. We also discussed data integration challenges and how processed data is prepared for AI model analysis. Understanding these processing steps is essential for working with the system effectively, which we'll explore further in the next chapter on risk assessment and prediction.