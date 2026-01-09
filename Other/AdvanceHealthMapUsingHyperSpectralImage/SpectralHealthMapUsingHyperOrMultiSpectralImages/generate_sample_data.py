import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

def generate_sample_hyperspectral_data(height=128, width=128, bands=224):
    """Generate realistic sample hyperspectral data"""
    print(f"Generating hyperspectral data: {height}x{width}x{bands}")
    
    # Create base reflectance pattern
    wavelengths = np.linspace(400, 2500, bands)
    data = np.zeros((bands, height, width))
    
    # Generate different crop health zones
    for i in range(height):
        for j in range(width):
            # Determine health status based on position
            distance_from_center = np.sqrt((i - height//2)**2 + (j - width//2)**2)
            
            if distance_from_center < 20:  # Healthy center
                signature = generate_healthy_signature(wavelengths)
            elif distance_from_center < 40:  # Moderate stress
                signature = generate_stressed_signature(wavelengths)
            elif distance_from_center < 50:  # Diseased areas
                signature = generate_diseased_signature(wavelengths)
            else:  # Healthy periphery
                signature = generate_healthy_signature(wavelengths) * 0.9
            
            # Add some noise
            noise = np.random.normal(0, 0.02, bands)
            data[:, i, j] = np.clip(signature + noise, 0, 1)
    
    # Add some disease spots
    disease_centers = [(30, 40), (80, 60), (90, 20)]
    for center in disease_centers:
        y, x = np.ogrid[:height, :width]
        mask = (y - center[0])**2 + (x - center[1])**2 <= 100
        
        for band in range(bands):
            diseased_value = generate_diseased_signature(wavelengths)[band]
            data[band, mask] = diseased_value + np.random.normal(0, 0.01)
    
    return data

def generate_healthy_signature(wavelengths):
    """Generate healthy vegetation spectral signature"""
    signature = np.zeros_like(wavelengths)
    
    # Visible region (low reflectance with chlorophyll absorption)
    visible_mask = wavelengths < 700
    signature[visible_mask] = 0.05 + 0.02 * np.random.random(np.sum(visible_mask))
    
    # Green peak around 550nm
    green_mask = (wavelengths >= 500) & (wavelengths <= 600)
    signature[green_mask] = 0.12 + 0.03 * np.random.random(np.sum(green_mask))
    
    # Red edge and NIR plateau (high reflectance)
    nir_mask = (wavelengths >= 700) & (wavelengths <= 1300)
    signature[nir_mask] = 0.45 + 0.05 * np.random.random(np.sum(nir_mask))
    
    # SWIR region (moderate reflectance with water absorption)
    swir1_mask = (wavelengths >= 1300) & (wavelengths <= 1800)
    signature[swir1_mask] = 0.35 + 0.03 * np.random.random(np.sum(swir1_mask))
    
    swir2_mask = wavelengths > 1800
    signature[swir2_mask] = 0.25 + 0.03 * np.random.random(np.sum(swir2_mask))
    
    return signature

def generate_stressed_signature(wavelengths):
    """Generate stressed vegetation spectral signature"""
    signature = generate_healthy_signature(wavelengths)
    
    # Reduce NIR reflectance (stress indicator)
    nir_mask = (wavelengths >= 700) & (wavelengths <= 1300)
    signature[nir_mask] *= 0.75
    
    # Increase visible reflectance
    visible_mask = wavelengths < 700
    signature[visible_mask] *= 1.2
    
    # Reduce green peak
    green_mask = (wavelengths >= 500) & (wavelengths <= 600)
    signature[green_mask] *= 0.8
    
    return signature

def generate_diseased_signature(wavelengths):
    """Generate diseased vegetation spectral signature"""
    signature = generate_healthy_signature(wavelengths)
    
    # Significantly reduce NIR reflectance
    nir_mask = (wavelengths >= 700) & (wavelengths <= 1300)
    signature[nir_mask] *= 0.5
    
    # Increase visible reflectance significantly
    visible_mask = wavelengths < 700
    signature[visible_mask] *= 1.6
    
    # Almost eliminate green peak
    green_mask = (wavelengths >= 500) & (wavelengths <= 600)
    signature[green_mask] *= 0.6
    
    return signature

def generate_environmental_data(num_days=30):
    """Generate realistic environmental sensor data"""
    print(f"Generating environmental data for {num_days} days")
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=num_days), 
                         end=datetime.now(), freq='H')
    
    # Generate realistic weather patterns
    data = []
    for i, date in enumerate(dates):
        hour_of_day = date.hour
        day_of_year = date.timetuple().tm_yday
        
        # Temperature with daily and seasonal cycles
        base_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        daily_temp_variation = 8 * np.sin(2 * np.pi * hour_of_day / 24)
        air_temp = base_temp + daily_temp_variation + np.random.normal(0, 2)
        soil_temp = air_temp - 2 + np.random.normal(0, 1)
        
        # Humidity (inverse correlation with temperature)
        humidity = 80 - (air_temp - 15) * 1.5 + np.random.normal(0, 5)
        humidity = np.clip(humidity, 20, 95)
        
        # Soil moisture (decreases over time without rain)
        base_moisture = 50 - (i % 168) * 0.2  # Weekly cycle
        soil_moisture = max(20, base_moisture + np.random.normal(0, 5))
        
        # Wind speed (higher during day)
        wind_speed = 2 + 3 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.exponential(1)
        wind_speed = max(0, wind_speed)
        
        # Rainfall (random events)
        rainfall = np.random.exponential(0.5) if np.random.random() < 0.1 else 0
        
        # Solar radiation (daily cycle, weather dependent)
        if 6 <= hour_of_day <= 18:
            solar_radiation = 800 * np.sin(np.pi * (hour_of_day - 6) / 12)
            solar_radiation *= (1 - rainfall / 10)  # Reduced during rain
        else:
            solar_radiation = 0
        solar_radiation = max(0, solar_radiation + np.random.normal(0, 50))
        
        # Leaf wetness (related to humidity and rainfall)
        leaf_wetness = min(100, humidity * 0.8 + rainfall * 10 + np.random.normal(0, 5))
        
        # CO2 level (slightly variable)
        co2_level = 410 + np.random.normal(0, 20)
        
        # pH level (relatively stable)
        ph_level = 6.5 + np.random.normal(0, 0.3)
        
        data.append({
            'timestamp': date,
            'air_temperature': round(air_temp, 1),
            'soil_temperature': round(soil_temp, 1),
            'humidity': round(humidity, 1),
            'soil_moisture': round(soil_moisture, 1),
            'wind_speed': round(wind_speed, 1),
            'rainfall': round(rainfall, 1),
            'solar_radiation': round(solar_radiation, 0),
            'leaf_wetness': round(leaf_wetness, 1),
            'co2_level': round(co2_level, 0),
            'ph_level': round(ph_level, 1)
        })
    
    return pd.DataFrame(data)

def generate_field_metadata():
    """Generate field metadata and crop information"""
    fields = {
        'field_a': {
            'name': 'Field A - Wheat',
            'crop_type': 'winter_wheat',
            'area_hectares': 120.5,
            'planting_date': '2024-03-15',
            'expected_harvest': '2024-08-30',
            'coordinates': {
                'center_lat': 41.8781,
                'center_lon': -87.6298,
                'bounds': [[41.870, -87.640], [41.886, -87.620]]
            },
            'soil_type': 'clay_loam',
            'irrigation_system': 'center_pivot',
            'management_zones': 4
        },
        'field_b': {
            'name': 'Field B - Corn',
            'crop_type': 'corn',
            'area_hectares': 85.2,
            'planting_date': '2024-04-20',
            'expected_harvest': '2024-09-15',
            'coordinates': {
                'center_lat': 41.8881,
                'center_lon': -87.6198,
                'bounds': [[41.880, -87.630], [41.896, -87.610]]
            },
            'soil_type': 'silt_loam',
            'irrigation_system': 'drip',
            'management_zones': 3
        },
        'field_c': {
            'name': 'Field C - Soybeans',
            'crop_type': 'soybeans',
            'area_hectares': 200.8,
            'planting_date': '2024-05-10',
            'expected_harvest': '2024-10-05',
            'coordinates': {
                'center_lat': 41.8681,
                'center_lon': -87.6398,
                'bounds': [[41.860, -87.650], [41.876, -87.630]]
            },
            'soil_type': 'sandy_loam',
            'irrigation_system': 'sprinkler',
            'management_zones': 5
        }
    }
    
    return fields

def create_sample_datasets():
    """Create complete sample datasets for demonstration"""
    print("🌱 Creating AI-Powered Spectral Health Mapping Sample Data")
    print("=" * 60)
    
    # Create data directory
    data_dir = "data/sample"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate field metadata
    print("📋 Generating field metadata...")
    fields_metadata = generate_field_metadata()
    
    import json
    with open(os.path.join(data_dir, "fields_metadata.json"), 'w') as f:
        json.dump(fields_metadata, f, indent=2)
    print("✅ Field metadata saved")
    
    # Generate hyperspectral data for each field
    for field_id, field_info in fields_metadata.items():
        print(f"\n🛰️ Generating hyperspectral data for {field_info['name']}")
        
        # Generate different sized data based on field
        if field_id == 'field_a':
            hyperspectral_data = generate_sample_hyperspectral_data(128, 128, 224)
        elif field_id == 'field_b':
            hyperspectral_data = generate_sample_hyperspectral_data(96, 96, 224)
        else:
            hyperspectral_data = generate_sample_hyperspectral_data(160, 160, 224)
        
        # Save hyperspectral data
        filename = f"{field_id}_hyperspectral.npy"
        filepath = os.path.join(data_dir, filename)
        np.save(filepath, hyperspectral_data)
        print(f"✅ Saved {filename} - Shape: {hyperspectral_data.shape}")
        
        # Generate and save environmental data
        env_data = generate_environmental_data(60)  # 60 days of data
        env_filename = f"{field_id}_environmental.csv"
        env_filepath = os.path.join(data_dir, env_filename)
        env_data.to_csv(env_filepath, index=False)
        print(f"✅ Saved {env_filename} - {len(env_data)} records")
    
    # Generate wavelength information
    print("\n📊 Generating wavelength calibration data...")
    wavelengths = np.linspace(400, 2500, 224)
    wavelength_info = {
        'wavelengths_nm': wavelengths.tolist(),
        'spectral_resolution': 'Variable (400-700nm: 2.5nm, 700-1000nm: 5nm, 1000-2500nm: 10nm)',
        'bands_count': 224,
        'calibration_date': datetime.now().isoformat(),
        'sensor_type': 'Hyperspectral_Imager_v2'
    }
    
    with open(os.path.join(data_dir, "wavelength_info.json"), 'w') as f:
        json.dump(wavelength_info, f, indent=2)
    print("✅ Wavelength calibration data saved")
    
    # Generate sample training labels for ML models
    print("\n🎯 Generating training labels...")
    training_labels = generate_training_labels()
    
    with open(os.path.join(data_dir, "training_labels.json"), 'w') as f:
        json.dump(training_labels, f, indent=2)
    print("✅ Training labels saved")
    
    # Create README file
    create_sample_data_readme(data_dir)
    
    print("\n🎉 Sample data generation completed!")
    print(f"📁 Data directory: {os.path.abspath(data_dir)}")
    print("\nGenerated files:")
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  📄 {file} ({file_size:.1f} MB)")

def generate_training_labels():
    """Generate sample training labels for machine learning models"""
    return {
        'health_classes': {
            0: 'healthy',
            1: 'mild_stress', 
            2: 'moderate_stress',
            3: 'severe_stress_or_disease'
        },
        'disease_types': {
            0: 'no_disease',
            1: 'fungal_infection',
            2: 'bacterial_disease',
            3: 'viral_infection',
            4: 'pest_damage'
        },
        'stress_types': {
            0: 'no_stress',
            1: 'water_stress',
            2: 'nutrient_deficiency',
            3: 'heat_stress',
            4: 'cold_stress'
        },
        'sample_annotations': {
            'field_a': {
                'healthy_regions': [[0, 0, 40, 40], [80, 80, 128, 128]],
                'diseased_regions': [[25, 35, 35, 45], [75, 55, 85, 65]],
                'stressed_regions': [[40, 40, 80, 80]]
            },
            'field_b': {
                'healthy_regions': [[0, 0, 30, 30], [60, 60, 96, 96]],
                'diseased_regions': [[20, 30, 30, 40]],
                'stressed_regions': [[30, 30, 60, 60]]
            },
            'field_c': {
                'healthy_regions': [[0, 0, 50, 50], [100, 100, 160, 160]],
                'diseased_regions': [[40, 60, 60, 80], [120, 40, 140, 60]],
                'stressed_regions': [[50, 50, 100, 100]]
            }
        },
        'vegetation_indices_thresholds': {
            'NDVI': {'healthy': [0.7, 1.0], 'stressed': [0.4, 0.7], 'diseased': [0.0, 0.4]},
            'EVI': {'healthy': [0.5, 0.8], 'stressed': [0.3, 0.5], 'diseased': [0.0, 0.3]},
            'SAVI': {'healthy': [0.6, 0.9], 'stressed': [0.3, 0.6], 'diseased': [0.0, 0.3]}
        }
    }

def create_sample_data_readme(data_dir):
    """Create README file explaining the sample data"""
    readme_content = """# AI-Powered Spectral Health Mapping - Sample Data

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

Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""

    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("📚 README.md created")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample datasets
    create_sample_datasets()
    
    print("\n🚀 Ready to run the AI-Powered Spectral Health Mapping System!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start dashboard: python main.py --mode dashboard")
    print("3. Or run analysis: python main.py --mode batch --data data/sample")