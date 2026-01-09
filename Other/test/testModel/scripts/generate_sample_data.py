"""
Script to generate sample .npy image data files for the crop health monitoring project.
This creates synthetic multispectral satellite imagery data.
"""

import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_npy_files():
    """Generate sample .npy files with synthetic multispectral imagery data."""
    
    # Create data/images directory if it doesn't exist
    os.makedirs('data/images', exist_ok=True)
    
    # Parameters for synthetic data generation
    height, width = 256, 256  # Image dimensions
    num_bands = 12  # Number of spectral bands (typical for satellite imagery)
    
    # Generate sample dates
    base_date = datetime(2021, 7, 14)
    dates = [base_date + timedelta(days=i) for i in range(100)]
    
    print("Generating sample .npy files...")
    
    for i, date in enumerate(dates, 1):
        # Generate synthetic multispectral data
        # Each band represents different wavelengths (RGB, NIR, SWIR, etc.)
        image_data = np.random.rand(height, width, num_bands).astype(np.float32)
        
        # Add some realistic patterns
        # Simulate vegetation indices patterns
        vegetation_mask = np.random.rand(height, width) > 0.3
        image_data[:, :, 3] = np.where(vegetation_mask, 
                                      np.random.uniform(0.6, 0.9, (height, width)),
                                      np.random.uniform(0.1, 0.4, (height, width)))
        
        # Simulate water bodies
        water_mask = np.random.rand(height, width) > 0.85
        image_data[:, :, 0:3] = np.where(water_mask[:, :, np.newaxis], 
                                        np.random.uniform(0.05, 0.15, (height, width, 3)),
                                        image_data[:, :, 0:3])
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, image_data.shape)
        image_data = np.clip(image_data + noise, 0, 1)
        
        # Save as .npy file
        filename = f"roi_{i:02d}_{date.strftime('%Y%m%d')}.npy"
        filepath = os.path.join('data', 'images', filename)
        np.save(filepath, image_data)
        
        print(f"Generated: {filename} - Shape: {image_data.shape}")
    
    print(f"\nGenerated {len(dates)} sample .npy files in data/images/")

if __name__ == "__main__":
    generate_sample_npy_files()