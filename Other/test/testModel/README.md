# Crop Health Monitoring with Hybrid AI System

A comprehensive machine learning system for crop health monitoring using satellite imagery and sensor data. This project combines CNN+LSTM deep learning models for spatial-temporal analysis with Random Forest ensemble methods for tabular sensor data processing.

## 🚀 Features

- **Hybrid AI Model**: Combines deep learning (CNN+LSTM) with traditional ML (Random Forest)
- **Multi-Modal Data Processing**: Handles both satellite imagery (.npy) and tabular sensor data
- **Spatial-Temporal Analysis**: Processes time series of satellite images for trend analysis
- **Health Classification**: Categorizes crop health into 5 levels (Excellent, Good, Fair, Poor, Critical)
- **Visual Health Maps**: Generates color-coded spatial health maps with vegetation indices
- **Comprehensive Prediction**: Real-time crop health prediction for new samples
- **Feature Importance Analysis**: Identifies key factors affecting crop health

## 📁 Project Structure

```
project/
├── data/
│   ├── images/                    # Satellite imagery (.npy files)
│   │   ├── roi_01_20210714.npy   # Multispectral image data
│   │   ├── roi_02_20210715.npy
│   │   └── ...
│   └── final_dataset.csv          # Tabular features + labels for training
│
├── scripts/
│   ├── train_hybrid_model.py      # Train CNN+LSTM and Random Forest ensemble
│   ├── predict_new_sample.py      # Make prediction on new image+sensor input
│   ├── visualize_health_map.py    # Generate spatial crop health maps
│   ├── generate_sample_data.py    # Generate synthetic sample data
│   └── generate_dataset.py        # Create sample tabular dataset
│
├── models/
│   ├── hybrid_cnn_lstm_model.h5   # Saved CNN+LSTM model
│   ├── rf_classifier.pkl          # Saved Random Forest ensemble
│   ├── label_encoder.pkl          # Label encoder for classes
│   └── scaler.pkl                 # StandardScaler for tabular features
│
├── outputs/
│   ├── healthmap_roi_001.png      # Example output maps
│   └── feature_importance.png     # Feature importance visualization
│
├── requirements.txt               # Required Python packages
└── README.md                      # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crop-health-monitoring
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv crop_health_env
   
   # Windows
   crop_health_env\Scripts\activate
   
   # Linux/Mac
   source crop_health_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Requirements

### Satellite Imagery (.npy files)
- **Format**: NumPy arrays saved as .npy files
- **Shape**: (Height, Width, Channels) - typically (256, 256, 12)
- **Channels**: Multispectral bands (Blue, Green, Red, NIR, SWIR, etc.)
- **Values**: Normalized reflectance values (0-1 range)
- **Naming**: `roi_XX_YYYYMMDD.npy` format

### Tabular Sensor Data
The CSV file should contain the following columns:

**Identifiers:**
- `roi_id`: Region of interest identifier
- `date`: Date of observation (YYYY-MM-DD)

**Weather/Environmental Features:**
- `temperature_avg`: Average temperature (°C)
- `humidity_avg`: Average humidity (%)
- `rainfall_mm`: Rainfall (mm)
- `wind_speed`: Wind speed (km/h)
- `solar_radiation`: Solar radiation (MJ/m²)

**Soil Features:**
- `soil_moisture`: Soil moisture (%)
- `soil_ph`: Soil pH level
- `soil_nitrogen`: Nitrogen content (ppm)
- `soil_phosphorus`: Phosphorus content (ppm)
- `soil_potassium`: Potassium content (ppm)

**Vegetation Indices:**
- `ndvi`: Normalized Difference Vegetation Index
- `evi`: Enhanced Vegetation Index
- `savi`: Soil Adjusted Vegetation Index
- `ndwi`: Normalized Difference Water Index
- `lai`: Leaf Area Index

**Management Features:**
- `days_since_planting`: Days since crop planting
- `irrigation_frequency`: Irrigation frequency per week
- `fertilizer_applied`: Binary (0/1)
- `pesticide_applied`: Binary (0/1)

**Spectral Features:**
- `band_X_mean`: Mean reflectance for band X
- `band_X_std`: Standard deviation for band X

**Target Variables:**
- `health_score`: Health score (0-100)
- `health_class`: Health category (Excellent/Good/Fair/Poor/Critical)

## 🚀 Quick Start

### 1. Generate Sample Data (Optional)
If you don't have real data, generate synthetic data for testing:

```bash
cd scripts
python generate_sample_data.py    # Generate .npy files
python generate_dataset.py        # Generate CSV dataset
```

### 2. Train the Model
```bash
cd scripts
python train_hybrid_model.py
```

This will:
- Load and preprocess the data
- Train Random Forest model on tabular features
- Build CNN+LSTM architecture for spatial-temporal analysis
- Save trained models to `models/` directory
- Generate feature importance plots

### 3. Make Predictions
```bash
cd scripts

# Predict from CSV data
python predict_new_sample.py --csv ../data/final_dataset.csv --roi_id roi_01 --date 2021-07-14

# Predict with image sequence and sensor data
python predict_new_sample.py --images ../data/images/roi_01_*.npy --csv ../data/final_dataset.csv
```

### 4. Generate Health Maps
```bash
cd scripts

# Single image health map
python visualize_health_map.py --input ../data/images/roi_01_20210714.npy

# Comparison map for multiple images
python visualize_health_map.py --input ../data/images/ --compare

# Run with sample data (no arguments)
python visualize_health_map.py
```

## 📈 Model Architecture

### CNN+LSTM Component
- **CNN Layers**: Extract spatial features from multispectral imagery
- **Time-Distributed**: Apply CNN to each time step in sequence
- **LSTM Layers**: Model temporal dependencies in satellite image sequences
- **Output**: 32-dimensional spatial-temporal feature vector

### Random Forest Component
- **Input**: 27 tabular sensor and environmental features
- **Architecture**: 200 trees with optimized hyperparameters
- **Output**: Health class probabilities and predictions

### Ensemble Strategy
The hybrid model combines both components for final predictions, leveraging:
- Spatial-temporal patterns from satellite imagery
- Environmental and sensor measurements
- Crop management practices
- Historical trends and seasonality

## 🎯 Health Classification

The system classifies crop health into 5 categories:

| Class | Score Range | Description | Color |
|-------|-------------|-------------|-------|
| **Excellent** | 80-100 | Optimal health, high productivity | 🟢 Green |
| **Good** | 65-79 | Good health, minor issues | 🟡 Light Green |
| **Fair** | 50-64 | Moderate health, requires attention | 🟡 Yellow |
| **Poor** | 35-49 | Poor health, significant issues | 🟠 Orange |
| **Critical** | 0-34 | Critical condition, immediate action needed | 🔴 Red |

## 📊 Output Examples

### Health Map Visualization
The system generates comprehensive health maps showing:
- RGB satellite imagery
- Vegetation indices (NDVI, EVI, SAVI)
- Health score distribution
- Color-coded health classification
- Statistical summaries

### Prediction Output
```json
{
    "final": {
        "predicted_class": "Good",
        "confidence": 0.847,
        "prediction_timestamp": "2024-01-15T10:30:00"
    },
    "tabular": {
        "class_probabilities": {
            "Excellent": 0.125,
            "Good": 0.847,
            "Fair": 0.028,
            "Poor": 0.000,
            "Critical": 0.000
        }
    },
    "ground_truth": {
        "actual_class": "Good",
        "actual_score": 72.3
    }
}
```

## 🔧 Configuration

### Model Parameters
Edit `train_hybrid_model.py` to modify:
- Image sequence length for LSTM
- CNN architecture (layers, filters)
- Random Forest hyperparameters
- Feature selection criteria

### Visualization Settings
Edit `visualize_health_map.py` to customize:
- Health class color schemes
- Vegetation indices weights
- Map layouts and styling
- Statistical displays

## 📋 API Reference

### Training
```python
from train_hybrid_model import HybridCropHealthModel

model = HybridCropHealthModel()
results = model.train(csv_path='data/final_dataset.csv')
```

### Prediction
```python
from predict_new_sample import CropHealthPredictor

predictor = CropHealthPredictor()
result = predictor.predict(sensor_data=sensor_dict)
```

### Visualization
```python
from visualize_health_map import CropHealthMapGenerator

generator = CropHealthMapGenerator()
health_map = generator.create_health_map('data/images/roi_01.npy')
```

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues with TensorFlow**
   ```bash
   pip install tensorflow-cpu  # Use CPU version
   ```

3. **Memory Issues with Large Images**
   - Reduce image dimensions in preprocessing
   - Process images in smaller batches
   - Use data generators for training

4. **Missing .npy Files**
   - Run `generate_sample_data.py` to create synthetic data
   - Check file paths and naming conventions
   - Ensure images have correct shape (H, W, C)

### Performance Optimization

1. **Training Speed**
   - Use GPU acceleration for CNN+LSTM training
   - Enable parallel processing in Random Forest (`n_jobs=-1`)
   - Reduce image resolution if needed

2. **Memory Usage**
   - Process images in batches
   - Use data generators instead of loading all data
   - Monitor memory usage with task manager

3. **Prediction Speed**
   - Pre-load models once for multiple predictions
   - Use batch prediction for multiple samples
   - Consider model quantization for deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Satellite imagery data processing techniques
- Vegetation indices calculation methods
- Deep learning architectures for remote sensing
- Agricultural monitoring best practices

## 📧 Contact

For questions, issues, or collaborations:
- Create an issue on GitHub
- Email: [your-email@domain.com]
- Project Link: [repository-url]

---

**Note**: This project is designed for research and educational purposes. For production deployment in agricultural monitoring systems, additional validation and testing with real-world data is recommended.