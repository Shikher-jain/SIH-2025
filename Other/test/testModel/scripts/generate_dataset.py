"""
Script to generate a sample final_dataset.csv with tabular features and labels for training.
This includes both image-derived features and sensor data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_dataset():
    """Generate sample dataset with tabular features and labels."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_samples = 10000
    
    # Generate sample dates
    base_date = datetime(2021, 7, 14)
    dates = [base_date + timedelta(days=np.random.randint(0, 100)) for _ in range(n_samples)]
    
    # Generate ROI identifiers
    roi_ids = [f"roi_{np.random.randint(1, 11):02d}" for _ in range(n_samples)]
    
    # Generate tabular features
    data = {
        'roi_id': roi_ids,
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        
        # Weather/Environmental features
        'temperature_avg': np.random.normal(25, 8, n_samples),  # Average temperature (°C)
        'humidity_avg': np.random.normal(65, 15, n_samples),    # Average humidity (%)
        'rainfall_mm': np.random.exponential(5, n_samples),     # Rainfall (mm)
        'wind_speed': np.random.gamma(2, 3, n_samples),         # Wind speed (km/h)
        'solar_radiation': np.random.normal(20, 5, n_samples),  # Solar radiation (MJ/m²)
        
        # Soil features
        'soil_moisture': np.random.beta(2, 2, n_samples) * 100,        # Soil moisture (%)
        'soil_ph': np.random.normal(6.5, 0.8, n_samples),              # Soil pH
        'soil_nitrogen': np.random.normal(50, 15, n_samples),           # Nitrogen (ppm)
        'soil_phosphorus': np.random.normal(30, 10, n_samples),         # Phosphorus (ppm)
        'soil_potassium': np.random.normal(200, 50, n_samples),         # Potassium (ppm)
        
        # Vegetation indices (derived from satellite imagery)
        'ndvi': np.random.beta(3, 2, n_samples),                        # NDVI (0-1)
        'evi': np.random.beta(2, 3, n_samples),                         # EVI (0-1)
        'savi': np.random.beta(2.5, 2.5, n_samples),                    # SAVI (0-1)
        'ndwi': np.random.normal(0.3, 0.2, n_samples),                  # NDWI (-1 to 1)
        'lai': np.random.gamma(2, 1.5, n_samples),                      # Leaf Area Index
        
        # Crop management features
        'days_since_planting': np.random.randint(0, 120, n_samples),    # Days since planting
        'irrigation_frequency': np.random.poisson(2, n_samples),        # Irrigation per week
        'fertilizer_applied': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # Binary
        'pesticide_applied': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),   # Binary
        
        # Spectral features (summary statistics from multispectral bands)
        'band_1_mean': np.random.normal(0.15, 0.05, n_samples),         # Blue band
        'band_2_mean': np.random.normal(0.18, 0.06, n_samples),         # Green band
        'band_3_mean': np.random.normal(0.12, 0.04, n_samples),         # Red band
        'band_4_mean': np.random.normal(0.35, 0.1, n_samples),          # NIR band
        'band_1_std': np.random.exponential(0.02, n_samples),           # Blue std
        'band_2_std': np.random.exponential(0.025, n_samples),          # Green std
        'band_3_std': np.random.exponential(0.02, n_samples),           # Red std
        'band_4_std': np.random.exponential(0.03, n_samples),           # NIR std
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate crop health labels based on feature combinations
    health_scores = []
    health_classes = []
    
    for idx, row in df.iterrows():
        # Calculate health score based on multiple factors
        score = 0.0
        
        # NDVI contribution (higher is better)
        score += row['ndvi'] * 30
        
        # Soil moisture contribution (optimal range 40-80%)
        if 40 <= row['soil_moisture'] <= 80:
            score += 20
        else:
            score += max(0, 20 - abs(row['soil_moisture'] - 60) * 0.5)
        
        # Temperature contribution (optimal range 18-28°C)
        if 18 <= row['temperature_avg'] <= 28:
            score += 15
        else:
            score += max(0, 15 - abs(row['temperature_avg'] - 23) * 0.8)
        
        # Rainfall contribution (moderate is better)
        if 2 <= row['rainfall_mm'] <= 8:
            score += 15
        else:
            score += max(0, 15 - abs(row['rainfall_mm'] - 5) * 2)
        
        # Soil pH contribution (optimal range 6.0-7.0)
        if 6.0 <= row['soil_ph'] <= 7.0:
            score += 10
        else:
            score += max(0, 10 - abs(row['soil_ph'] - 6.5) * 8)
        
        # LAI contribution
        score += min(row['lai'] * 5, 10)
        
        # Add some random variation
        score += np.random.normal(0, 5)
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        health_scores.append(score)
        
        # Convert to categorical labels
        if score >= 80:
            health_classes.append('Excellent')
        elif score >= 65:
            health_classes.append('Good')
        elif score >= 50:
            health_classes.append('Fair')
        elif score >= 35:
            health_classes.append('Poor')
        else:
            health_classes.append('Critical')
    
    # Add health labels to DataFrame
    df['health_score'] = health_scores
    df['health_class'] = health_classes
    
    # Reorder columns
    columns_order = ['roi_id', 'date', 'health_score', 'health_class'] + \
                   [col for col in df.columns if col not in ['roi_id', 'date', 'health_score', 'health_class']]
    df = df[columns_order]
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/final_dataset.csv', index=False)
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nHealth class distribution:")
    print(df['health_class'].value_counts())
    print(f"\nDataset saved to: data/final_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Display sample of the data
    print(f"\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    generate_dataset()