"""
Generate an updated dataset CSV that matches your existing code requirements.
This creates a dataset with the required columns for the hybrid model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_updated_dataset():
    """Generate dataset with hyperspectral-specific features."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_samples = 5000
    
    # Generate image names matching the generated files
    image_names = [f"roi_{i:03d}" for i in range(1, 11)] * (n_samples // 10 + 1)
    image_names = image_names[:n_samples]
    
    # Generate sample dates
    base_date = datetime(2021, 7, 14)
    dates = [base_date + timedelta(days=np.random.randint(0, 100)) for _ in range(n_samples)]
    
    # Generate hyperspectral-derived indices
    data = {
        'ImageName': image_names,
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        
        # Hyperspectral vegetation indices
        'NDVI': np.random.beta(3, 2, n_samples),           # NDVI (0-1)
        'NDWI': np.random.normal(0.3, 0.2, n_samples),     # NDWI (-1 to 1)
        'NDSI': np.random.normal(0.2, 0.15, n_samples),    # NDSI (-1 to 1)
        'CHI': np.random.beta(2.5, 2, n_samples),          # Chlorophyll Index
        'GNDVI': np.random.beta(2, 2, n_samples),          # Green NDVI (0-1)
        
        # Environmental/Sensor features  
        'SoilMoisture': np.random.beta(2, 2, n_samples),   # Soil moisture (0-1)
        'AirTemp': np.random.normal(25, 8, n_samples),     # Air temperature (°C)
        'Humidity': np.random.normal(65, 15, n_samples),   # Humidity (%)
        'Rainfall': np.random.exponential(5, n_samples),   # Rainfall (mm)
        'pH': np.random.normal(6.5, 0.8, n_samples),       # Soil pH
        
        # Additional features for context
        'elevation': np.random.normal(500, 200, n_samples),      # Elevation (m)
        'slope': np.random.exponential(5, n_samples),            # Slope (degrees)
        'aspect': np.random.uniform(0, 360, n_samples),          # Aspect (degrees)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['NDVI'] = np.clip(df['NDVI'], 0, 1)
    df['NDWI'] = np.clip(df['NDWI'], -1, 1)
    df['NDSI'] = np.clip(df['NDSI'], -1, 1)
    df['CHI'] = np.clip(df['CHI'], 0, 1)
    df['GNDVI'] = np.clip(df['GNDVI'], 0, 1)
    df['SoilMoisture'] = np.clip(df['SoilMoisture'], 0, 1)
    df['AirTemp'] = np.clip(df['AirTemp'], -10, 50)
    df['Humidity'] = np.clip(df['Humidity'], 0, 100)
    df['pH'] = np.clip(df['pH'], 4, 9)
    df['elevation'] = np.clip(df['elevation'], 0, 3000)
    df['slope'] = np.clip(df['slope'], 0, 45)
    
    # Generate labels based on multiple factors
    labels = []
    
    for idx, row in df.iterrows():
        # Calculate health score based on indices
        health_score = 0.0
        
        # NDVI contribution (primary factor)
        health_score += row['NDVI'] * 40
        
        # Soil moisture contribution (optimal range 0.3-0.7)
        if 0.3 <= row['SoilMoisture'] <= 0.7:
            health_score += 25
        else:
            health_score += max(0, 25 - abs(row['SoilMoisture'] - 0.5) * 50)
        
        # Temperature contribution (optimal range 18-28°C)
        if 18 <= row['AirTemp'] <= 28:
            health_score += 20
        else:
            health_score += max(0, 20 - abs(row['AirTemp'] - 23) * 1.5)
        
        # Humidity contribution (optimal range 50-80%)
        if 50 <= row['Humidity'] <= 80:
            health_score += 10
        else:
            health_score += max(0, 10 - abs(row['Humidity'] - 65) * 0.2)
        
        # pH contribution (optimal range 6.0-7.0)
        if 6.0 <= row['pH'] <= 7.0:
            health_score += 5
        else:
            health_score += max(0, 5 - abs(row['pH'] - 6.5) * 3)
        
        # Add some random variation
        health_score += np.random.normal(0, 3)
        
        # Ensure score is within bounds
        health_score = max(0, min(100, health_score))
        
        # Convert to simplified labels (matching your existing code)
        if health_score >= 70:
            labels.append('Healthy')
        elif health_score >= 40:
            labels.append('Stressed')
        else:
            labels.append('Diseased')
    
    # Add labels to DataFrame
    df['Label'] = labels
    
    # Reorder columns
    columns_order = ['ImageName', 'date', 'Label'] + [col for col in df.columns if col not in ['ImageName', 'date', 'Label']]
    df = df[columns_order]
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/final_dataset.csv', index=False)
    
    print(f"Generated updated dataset with {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df['Label'].value_counts())
    print(f"\nDataset saved to: data/final_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Display sample of the data
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    generate_updated_dataset()