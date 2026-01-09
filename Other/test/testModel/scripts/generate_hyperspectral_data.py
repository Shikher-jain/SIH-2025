"""
Generate hyperspectral sample data matching your existing code requirements.
This creates .npy files with shape (64, 64, 50) for hyperspectral analysis.
"""

import numpy as np
import os
from datetime import datetime, timedelta

def generate_hyperspectral_npy_files():
    """Generate hyperspectral .npy files with 50 spectral bands."""
    
    # Create data/images directory if it doesn't exist
    os.makedirs('data/images', exist_ok=True)
    
    # Parameters for hyperspectral data
    height, width = 64, 64
    num_bands = 50  # Hyperspectral bands
    
    # Generate sample dates and ROI names
    base_date = datetime(2021, 7, 14)
    dates = [base_date + timedelta(days=i) for i in range(100)]
    
    print("Generating hyperspectral .npy files...")
    
    for i, date in enumerate(dates, 1):
        # Generate realistic hyperspectral data
        image_data = np.random.rand(height, width, num_bands).astype(np.float32)
        
        # Add realistic spectral patterns
        for band in range(num_bands):
            # Simulate different land cover types
            wavelength_factor = (band + 1) / num_bands
            
            # Vegetation signature (higher reflectance in NIR)
            vegetation_mask = np.random.rand(height, width) > 0.4
            if band > 25:  # NIR region
                image_data[:, :, band] = np.where(vegetation_mask, 
                                                 np.random.uniform(0.6, 0.9, (height, width)),
                                                 image_data[:, :, band])
            elif 15 <= band <= 25:  # Red region
                image_data[:, :, band] = np.where(vegetation_mask,
                                                 np.random.uniform(0.1, 0.3, (height, width)),
                                                 image_data[:, :, band])
            
            # Water signature (low reflectance across all bands)
            water_mask = np.random.rand(height, width) > 0.85
            image_data[:, :, band] = np.where(water_mask,
                                            np.random.uniform(0.02, 0.1, (height, width)),
                                            image_data[:, :, band])
            
            # Soil signature (gradual increase with wavelength)
            soil_mask = ~vegetation_mask & ~water_mask
            if soil_mask.any():
                soil_reflectance = 0.2 + 0.3 * wavelength_factor
                image_data[:, :, band] = np.where(soil_mask,
                                                np.random.normal(soil_reflectance, 0.05, (height, width)),
                                                image_data[:, :, band])
        
        # Ensure values are in valid range
        image_data = np.clip(image_data, 0, 1)
        
        # Add some noise
        noise = np.random.normal(0, 0.02, image_data.shape)
        image_data = np.clip(image_data + noise, 0, 1)
        
        # Save as .npy file
        filename = f"roi_{i:03d}.npy"
        filepath = os.path.join('data', 'images', filename)
        np.save(filepath, image_data)
        
        print(f"Generated: {filename} - Shape: {image_data.shape}")
    
    print(f"\nGenerated {len(dates)} hyperspectral .npy files in data/images/")
    print("Each file contains 64x64x50 hyperspectral data suitable for 3D CNN processing.")

if __name__ == "__main__":
    generate_hyperspectral_npy_files()