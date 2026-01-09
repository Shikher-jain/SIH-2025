#!/usr/bin/env python3
"""
Test script to process just the first coordinate and verify 256x256 output
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

def initialize_earth_engine(service_account_path="../earth-engine-service-account.json"):
    """Initialize Google Earth Engine with service account authentication"""
    try:
        with open(service_account_path, 'r') as f:
            service_account = json.load(f)
        
        print(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")
        
        credentials = ee.ServiceAccountCredentials(
            service_account['client_email'], 
            service_account_path
        )
        ee.Initialize(credentials)
        print("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

def load_config(config_path="./config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def main():
    print("Testing 256x256 Output Generation")
    print("="*40)
    
    # Load config
    config = load_config()
    if not config:
        return
    
    # Initialize GEE
    if not initialize_earth_engine():
        return
    
    # Get first coordinate
    features = config.get('area', {}).get('geojson', {}).get('features', [])
    if not features:
        print("No features found")
        return
    
    first_feature = features[0]
    coordinates = first_feature['geometry']['coordinates']
    
    # Create polygon
    polygon = ee.Geometry.Polygon(coordinates)
    
    # Search for image
    start_date = '2018-02-01'
    end_date = '2018-04-30'
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(polygon)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    if collection.size().getInfo() == 0:
        print("No images found")
        return
    
    image = collection.first()
    rgb_image = image.select(['B4', 'B3', 'B2']).rename(['Red', 'Green', 'Blue'])
    
    # Export with fixed 256x256 dimensions
    bounds = polygon.bounds().getInfo()
    coords = bounds['coordinates'][0]
    
    min_lon = min(coord[0] for coord in coords)
    max_lon = max(coord[0] for coord in coords)
    min_lat = min(coord[1] for coord in coords)
    max_lat = max(coord[1] for coord in coords)
    
    # Fixed 256x256 dimensions
    width = 256
    height = 256
    
    scale_x = (max_lon - min_lon) / width
    scale_y = (max_lat - min_lat) / height
    
    request = {
        'expression': rgb_image.clip(polygon),
        'fileFormat': 'NUMPY_NDARRAY',
        'bandIds': ['Red', 'Green', 'Blue'],
        'grid': {
            'dimensions': {
                'width': width,
                'height': height
            },
            'affineTransform': {
                'scaleX': scale_x,
                'shearX': 0,
                'translateX': min_lon,
                'shearY': 0,
                'scaleY': -scale_y,
                'translateY': max_lat
            },
            'crsCode': 'EPSG:4326'
        }
    }
    
    print(f"Fetching {width}x{height} pixel data...")
    pixel_data = ee.data.computePixels(request)
    
    if pixel_data is not None:
        print(f"✅ Successfully fetched data with shape: {pixel_data.shape}")
        
        # Create simple grayscale heatmap
        if hasattr(pixel_data, 'dtype') and pixel_data.dtype.names is not None:
            red = pixel_data['Red'].astype(np.float32)
            green = pixel_data['Green'].astype(np.float32)
            blue = pixel_data['Blue'].astype(np.float32)
            
            # Convert to grayscale
            grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
            
            # Normalize
            data_min = np.nanmin(grayscale)
            data_max = np.nanmax(grayscale)
            
            if data_max > data_min:
                normalized = (grayscale - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(grayscale)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red to green
            cmap = LinearSegmentedColormap.from_list("RedToGreen", colors)
            
            plt.imshow(normalized, cmap=cmap, origin='lower')
            plt.colorbar(label='Normalized Values (Red=Low, Green=High)')
            plt.title(f'Test 256x256 Satellite Heatmap - Coordinate 1')
            plt.xlabel('X Pixels')
            plt.ylabel('Y Pixels')
            
            # Save
            os.makedirs('Test_Output', exist_ok=True)
            output_path = 'Test_Output/test_256x256_heatmap.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved 256x256 heatmap to: {output_path}")
            print(f"Image dimensions: {normalized.shape}")
        else:
            print("❌ Unexpected data format")
    else:
        print("❌ Failed to fetch data")

if __name__ == "__main__":
    main()