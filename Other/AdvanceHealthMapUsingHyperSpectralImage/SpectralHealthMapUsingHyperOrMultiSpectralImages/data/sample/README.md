# AI-Powered Spectral Health Mapping - Sample Data

This directory contains sample datasets for demonstrating the AI-powered spectral health mapping system.

## Data Files

### Hyperspectral Data
- `field_a_hyperspectral.npy`: Wheat field hyperspectral data (128x128x224)
- `field_b_hyperspectral.npy`: Corn field hyperspectral data (96x96x224)  
- `field_c_hyperspectral.npy`: Soybean field hyperspectral data (160x160x224)

Each file contains a 3D numpy array with dimensions [bands, height, width].
Spectral range: 400-2500 nm with 224 bands.

### Environmental Data
- `field_a_environmental.csv`: Environmental sensor data for wheat field
- `field_b_environmental.csv`: Environmental sensor data for corn field
- `field_c_environmental.csv`: Environmental sensor data for soybean field

Each CSV contains hourly measurements for:
- Air temperature (°C)
- Soil temperature (°C) 
- Humidity (%)
- Soil moisture (%)
- Wind speed (m/s)
- Rainfall (mm)
- Solar radiation (W/m²)
- Leaf wetness (%)
- CO2 level (ppm)
- pH level

### Metadata Files
- `fields_metadata.json`: Field information including crop types, coordinates, management zones
- `wavelength_info.json`: Spectral band calibration information
- `training_labels.json`: Sample labels for machine learning model training

## Health Zones in Sample Data

The hyperspectral data includes simulated health zones:

1. **Healthy zones**: High NDVI, strong NIR plateau, typical vegetation signature
2. **Stressed zones**: Reduced NIR reflectance, increased visible reflectance
3. **Diseased zones**: Significantly altered spectral signature, low vegetation indices
4. **Problem spots**: Scattered disease/pest damage areas

## Usage

Load the data using the main system:

```python
from main import SpectralHealthSystem

# Initialize system
system = SpectralHealthSystem()

# Run single field analysis
python main.py --mode single --data data/sample/field_a_hyperspectral.npy

# Run batch analysis on all fields
python main.py --mode batch --data data/sample

# Start interactive dashboard
python main.py --mode dashboard
```

## AI Model Training

Use the training labels to train custom models:

```python
# Load training annotations
import json
with open('training_labels.json', 'r') as f:
    labels = json.load(f)

# Use annotations for supervised learning
annotations = labels['sample_annotations']
health_classes = labels['health_classes']
```

## Real-World Adaptation

To use this system with real data:

1. Replace sample hyperspectral files with your sensor data
2. Update wavelength calibration in `wavelength_info.json`
3. Modify field metadata to match your locations
4. Retrain models with your ground truth data
5. Adjust vegetation index thresholds for your crops/region

Generated on: 2025-10-03 01:35:57
