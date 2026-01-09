#!/usr/bin/env python3
"""
Test script to verify satellite search functionality in test18
"""

import ee
import json
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Configuration from test18
DATE_RANGE_START = '2017-10-01'
DATE_RANGE_END = '2018-03-31'
CLOUD_THRESHOLD = 20
MAX_PIXELS = 1e10
SERVICE_ACCOUNT_PATH = '../earth-engine-service-account.json'

def initialize_earth_engine():
    """Initialize Google Earth Engine with service account authentication"""
    try:
        # Load service account credentials
        with open(SERVICE_ACCOUNT_PATH, 'r') as f:
            service_account = json.load(f)
        
        print(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")
        
        # Authenticate and initialize
        credentials = ee.ServiceAccountCredentials(
            service_account['client_email'], 
            SERVICE_ACCOUNT_PATH
        )
        ee.Initialize(credentials)
        print("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

def create_geometry_from_coordinates(coordinates):
    """Create GEE geometry from coordinate list"""
    try:
        return ee.Geometry.Polygon(coordinates)
    except Exception as e:
        print(f"Error creating geometry: {e}")
        return None

def search_satellite_image(polygon, start_date, end_date, cloud_threshold=20):
    """Search for satellite image in GEE for given polygon and date range"""
    try:
        print(f"Searching for Sentinel-2 images from {start_date} to {end_date} with cloud threshold {cloud_threshold}%")
        
        # Search for Sentinel-2 imagery (using HARMONIZED version)
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        # Check collection size
        size = collection.size().getInfo()
        print(f"Found {size} images in collection")
        
        if size == 0:
            print("No images found with current filters, trying without cloud filter...")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(polygon)
                         .filterDate(start_date, end_date)
                         .sort('CLOUDY_PIXEL_PERCENTAGE'))
            size = collection.size().getInfo()
            print(f"Found {size} images without cloud filter")
            
            if size == 0:
                # Try with extended date ranges (buffer periods)
                print("No images found, trying with buffer periods...")
                
                # Try with a broader date range (extend by 3 months on each side)
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    
                    # Extend start date by 3 months back
                    extended_start = start_dt - relativedelta(months=3)
                    extended_start_str = extended_start.strftime('%Y-%m-%d')
                    
                    # Extend end date by 3 months forward
                    extended_end = end_dt + relativedelta(months=3)
                    extended_end_str = extended_end.strftime('%Y-%m-%d')
                    
                    print(f"Trying extended date range: {extended_start_str} to {extended_end_str}")
                    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                 .filterBounds(polygon)
                                 .filterDate(extended_start_str, extended_end_str)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
                    
                    size = collection.size().getInfo()
                    print(f"Found {size} images in extended date range")
                    
                    if size == 0:
                        # Try with no cloud filter in extended range
                        print("No images found with cloud filter in extended range, trying without cloud filter...")
                        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                     .filterBounds(polygon)
                                     .filterDate(extended_start_str, extended_end_str)
                                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
                        size = collection.size().getInfo()
                        print(f"Found {size} images in extended range without cloud filter")
                        
                        if size == 0:
                            print("No images found even with extended date range and no cloud filter")
                            return None
                except Exception as date_error:
                    print(f"Error with date extension: {date_error}")
                    return None
        
        # Get the first (least cloudy) image
        image = collection.first()
        
        # Check if image exists
        info = image.getInfo()
        if info and 'id' in info:
            print(f"Selected image: {info['id']}")
            return image
        else:
            print("No suitable image found")
            return None
    except Exception as e:
        print(f"Error searching for satellite image: {e}")
        return None

def test_satellite_search():
    """Test the satellite search functionality with a sample polygon"""
    # Sample coordinates from test18 config (first feature)
    coordinates = [
        [
            [74.87309532655874, 31.34011708813776],
            [74.87309532655874, 31.325769593973476],
            [74.89499758710872, 31.325769593973476],
            [74.89499758710872, 31.34011708813776],
            [74.87309532655874, 31.34011708813776]
        ]
    ]
    
    print("Testing satellite search functionality...")
    print("=" * 50)
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        print("Failed to initialize Earth Engine")
        return
    
    # Create geometry
    polygon = create_geometry_from_coordinates(coordinates[0])
    if polygon is None:
        print("Failed to create polygon")
        return
    
    # Search for satellite image
    image = search_satellite_image(polygon, DATE_RANGE_START, DATE_RANGE_END, CLOUD_THRESHOLD)
    if image is None:
        print("No image found")
        return
    
    print("✅ Satellite search test completed successfully")

if __name__ == "__main__":
    test_satellite_search()