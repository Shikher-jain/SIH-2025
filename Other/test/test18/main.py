#!/usr/bin/env python3
"""
Test18 - GeoJSON Area Processing Pipeline

Processes multiple areas from GeoJSON coordinates one by one with the following features:
- Processes each area coordinate individually
- Minimizes pixel loss within GEE thresholds
- Outputs to structured directory format
- Uses date range Oct 2017 - Jan 2018
- Uses same 5 sensors as test15
- 50% cloud threshold
- Skips areas with no data or size issues without creating output folders
"""

import ee
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATE_RANGE_START = '2017-10-01'
DATE_RANGE_END = '2018-03-31'  # Extended to match test16
CLOUD_THRESHOLD = 20  # Reduced to match test16 (20%)
MAX_PIXELS = 1e10  # Increased to match test16
OUTPUT_BASE_DIR = 'test18_output'
SERVICE_ACCOUNT_PATH = './earth-engine-service-account.json'

# Sensor assets (same as test15)
SENSOR_ASSETS = {
    'ECe': 'projects/pk07007/assets/ECe',
    'N': 'projects/pk07007/assets/N',
    'P': 'projects/pk07007/assets/P',
    'pH': 'projects/pk07007/assets/pH',
    'OC': 'projects/pk07007/assets/OC'
}

def initialize_earth_engine():
    """Initialize Google Earth Engine with service account authentication"""
    try:
        # Load service account credentials
        with open(SERVICE_ACCOUNT_PATH, 'r') as f:
            service_account = json.load(f)
        
        logger.info(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")
        
        # Authenticate and initialize
        credentials = ee.ServiceAccountCredentials(
            service_account['client_email'], 
            SERVICE_ACCOUNT_PATH
        )
        ee.Initialize(credentials)
        logger.info("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def create_geometry_from_coordinates(coordinates):
    """Create GEE geometry from coordinate list"""
    try:
        return ee.Geometry.Polygon(coordinates)
    except Exception as e:
        logger.error(f"Error creating geometry: {e}")
        return None

def search_satellite_image(polygon, start_date, end_date, cloud_threshold=20):
    """Search for satellite image in GEE for given polygon and date range - adapted from test16"""
    try:
        logger.info(f"Searching for Sentinel-2 images from {start_date} to {end_date} with cloud threshold {cloud_threshold}%")
        
        # Search for Sentinel-2 imagery (using HARMONIZED version) - same as test16
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        # Check collection size
        size = collection.size().getInfo()
        logger.info(f"Found {size} images in collection")
        
        if size == 0:
            logger.info("No images found with current filters, trying without cloud filter...")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(polygon)
                         .filterDate(start_date, end_date)
                         .sort('CLOUDY_PIXEL_PERCENTAGE'))
            size = collection.size().getInfo()
            logger.info(f"Found {size} images without cloud filter")
            
            if size == 0:
                # Try with extended date ranges (buffer periods) - same approach as test16
                logger.info("No images found, trying with buffer periods...")
                
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
                    
                    logger.info(f"Trying extended date range: {extended_start_str} to {extended_end_str}")
                    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                 .filterBounds(polygon)
                                 .filterDate(extended_start_str, extended_end_str)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
                    
                    size = collection.size().getInfo()
                    logger.info(f"Found {size} images in extended date range")
                    
                    if size == 0:
                        # Try with no cloud filter in extended range
                        logger.info("No images found with cloud filter in extended range, trying without cloud filter...")
                        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                     .filterBounds(polygon)
                                     .filterDate(extended_start_str, extended_end_str)
                                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
                        size = collection.size().getInfo()
                        logger.info(f"Found {size} images in extended range without cloud filter")
                        
                        if size == 0:
                            logger.info("No images found even with extended date range and no cloud filter")
                            return None
                except Exception as date_error:
                    logger.error(f"Error with date extension: {date_error}")
                    return None
        
        # Get the first (least cloudy) image
        image = collection.first()
        
        # Check if image exists
        info = image.getInfo()
        if info and 'id' in info:
            logger.info(f"Selected image: {info['id']}")
            return image
        else:
            logger.info("No suitable image found")
            return None
    except Exception as e:
        logger.error(f"Error searching for satellite image: {e}")
        return None

def select_rgb_bands(image):
    """Select RGB bands (B4, B3, B2) from Sentinel-2 image"""
    try:
        # Select RGB bands: B4 (Red), B3 (Green), B2 (Blue)
        rgb_image = image.select(['B4', 'B3', 'B2']).rename(['Red', 'Green', 'Blue'])
        return rgb_image
    except Exception as e:
        logger.error(f"Error selecting RGB bands: {e}")
        return None

def export_image_data(image, region, scale=10):
    """Export image data as numpy array with specified scale (meters per pixel) - adapted from test16"""
    try:
        logger.info(f"Exporting image data with scale {scale} meters per pixel...")
        
        # Clip the image to the region
        clipped = image.clip(region)
        
        # Get the bounding box of the region
        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        # Calculate bounding box
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        # Calculate width and height in pixels based on scale
        # Earth's circumference is about 40,075,017 meters at the equator
        # 1 degree of longitude at equator ≈ 111,319 meters
        # 1 degree of latitude ≈ 111,139 meters (varies slightly with latitude)
        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139
        
        # Calculate dimensions in pixels
        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)
        
        # Limit to max pixels to prevent errors
        if width * height > MAX_PIXELS:
            logger.warning(f"Image size ({width}x{height} = {width * height}) exceeds GEE limit ({MAX_PIXELS})")
            # Reduce dimensions while maintaining aspect ratio
            ratio = width / height
            new_width = int(np.sqrt(MAX_PIXELS * ratio))
            new_height = int(MAX_PIXELS / new_width)
            width, height = new_width, new_height
            logger.info(f"Reduced dimensions to {width}x{height}")
        
        logger.info(f"Calculated image dimensions: {width}x{height} pixels")
        
        # Define grid parameters based on scale
        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height
        
        # Create request for computePixels
        request = {
            'expression': clipped,
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
                    'scaleY': -scale_y,  # Negative for correct orientation
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        # Fetch pixel data as numpy array
        logger.info("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)
        
        if pixel_data is not None:
            logger.info(f"Successfully fetched pixel data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch pixel data")
            return None
            
    except Exception as e:
        logger.error(f"Error exporting image data: {e}")
        return None

def get_sensor_data(region):
    """Get sensor data for all 5 sensors - adapted from test15 approach"""
    try:
        logger.info("Fetching sensor data...")
        sensor_data = {}
        valid_sensors = []
        
        for sensor_name, asset_id in SENSOR_ASSETS.items():
            try:
                logger.info(f"Loading {sensor_name} sensor data from {asset_id}...")
                sensor_image = ee.Image(asset_id)
                
                # Get image info to check if asset exists
                image_info = sensor_image.getInfo()
                if not image_info:
                    logger.warning(f"Failed to load {sensor_name}: No image info")
                    continue
                
                # Get band configuration - use the actual bands from the asset
                available_bands = image_info.get('bands', [])
                if len(available_bands) == 0:
                    logger.warning(f"No bands available for {sensor_name}")
                    continue
                
                # Select all available bands for this sensor (up to 4)
                actual_band_ids = [band['id'] for band in available_bands[:4]]
                logger.info(f"Available bands for {sensor_name}: {actual_band_ids}")
                
                # Select bands
                selected_image = sensor_image.select(actual_band_ids)
                valid_sensors.append(selected_image)
                sensor_data[sensor_name] = {
                    'bands': actual_band_ids,
                    'description': f"{sensor_name} sensor data"
                }
                
                logger.info(f"✅ {sensor_name}: {len(actual_band_ids)} bands loaded")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {sensor_name}: {e}")
                continue
        
        if len(valid_sensors) == 0:
            logger.error("No sensor data could be loaded")
            return None
            
        # Combine all sensor images
        combined_sensor_image = ee.Image.cat(valid_sensors)
        
        # Resample from native resolution to 10m
        combined_sensor_image = combined_sensor_image.reproject('EPSG:4326', scale=10).resample('bilinear')
        
        # Clip to region
        combined_sensor_image = combined_sensor_image.clip(region)
        
        logger.info(f"✅ Combined sensor data: {len(valid_sensors)} sensors loaded")
        return combined_sensor_image
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        return None

def calculate_ndvi(image):
    """Calculate NDVI from Sentinel-2 image"""
    try:
        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        # For Sentinel-2: NIR = B8, Red = B4
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi
    except Exception as e:
        logger.error(f"Error calculating NDVI: {e}")
        return None

def export_ndvi_data(ndvi_image, region, scale=10):
    """Export NDVI data as numpy array"""
    try:
        logger.info("Exporting NDVI data...")
        
        # Get the bounding box of the region
        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        # Calculate bounding box
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        # Calculate dimensions in pixels
        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139
        
        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)
        
        # Limit to max pixels
        if width * height > MAX_PIXELS:
            ratio = width / height
            new_width = int(np.sqrt(MAX_PIXELS * ratio))
            new_height = int(MAX_PIXELS / new_width)
            width, height = new_width, new_height
            logger.info(f"Reduced NDVI dimensions to {width}x{height}")
        
        logger.info(f"NDVI image dimensions: {width}x{height} pixels")
        
        # Create request for computePixels
        request = {
            'expression': ndvi_image,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': ['NDVI'],
            'grid': {
                'dimensions': {
                    'width': width,
                    'height': height
                },
                'affineTransform': {
                    'scaleX': (max_lon - min_lon) / width,
                    'shearX': 0,
                    'translateX': min_lon,
                    'shearY': 0,
                    'scaleY': -((max_lat - min_lat) / height),
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        # Fetch pixel data as numpy array
        pixel_data = ee.data.computePixels(request)
        
        if pixel_data is not None:
            logger.info(f"Successfully fetched NDVI data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch NDVI data")
            return None
            
    except Exception as e:
        logger.error(f"Error exporting NDVI data: {e}")
        return None

def export_sensor_data(image, region, scale=10):
    """Export sensor data as numpy array with specified scale (meters per pixel)"""
    try:
        logger.info(f"Exporting sensor data with scale {scale} meters per pixel...")
        
        # Clip the image to the region
        clipped = image.clip(region)
        
        # Get the bounding box of the region
        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        # Calculate bounding box
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        # Calculate width and height in pixels based on scale
        # Earth's circumference is about 40,075,017 meters at the equator
        # 1 degree of longitude at equator ≈ 111,319 meters
        # 1 degree of latitude ≈ 111,139 meters (varies slightly with latitude)
        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139
        
        # Calculate dimensions in pixels
        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)
        
        # Limit to max pixels to prevent errors
        if width * height > MAX_PIXELS:
            logger.warning(f"Image size ({width}x{height} = {width * height}) exceeds GEE limit ({MAX_PIXELS})")
            # Reduce dimensions while maintaining aspect ratio
            ratio = width / height
            new_width = int(np.sqrt(MAX_PIXELS * ratio))
            new_height = int(MAX_PIXELS / new_width)
            width, height = new_width, new_height
            logger.info(f"Reduced dimensions to {width}x{height}")
        
        logger.info(f"Calculated image dimensions: {width}x{height} pixels")
        
        # Define grid parameters based on scale
        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height
        
        # Get all band names from the image
        image_info = image.getInfo()
        band_names = [band['id'] for band in image_info.get('bands', [])]
        
        if not band_names:
            logger.error("No bands found in sensor image")
            return None
            
        logger.info(f"Sensor data bands: {band_names}")
        
        # Create request for computePixels
        request = {
            'expression': clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': band_names,
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
                    'scaleY': -scale_y,  # Negative for correct orientation
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        # Fetch pixel data as numpy array
        logger.info("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)
        
        if pixel_data is not None:
            logger.info(f"Successfully fetched sensor data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch sensor data")
            return None
            
    except Exception as e:
        logger.error(f"Error exporting sensor data: {e}")
        return None

def combine_ndvi_sensor_data(ndvi_data, sensor_data):
    """Combine NDVI and sensor data into a single 3D array with 21 bands and crop/pad to required shape"""
    try:
        logger.info("Combining NDVI and sensor data into 21-band array...")
        
        # Ensure both arrays have the same spatial dimensions
        ndvi_shape = ndvi_data.shape
        sensor_shape = sensor_data.shape
        
        logger.info(f"NDVI shape: {ndvi_shape}, Sensor shape: {sensor_shape}")
        
        # For structured arrays, we need to handle differently
        if hasattr(ndvi_data, 'dtype') and ndvi_data.dtype.names is not None:
            # NDVI is structured array, extract the NDVI band
            ndvi_values = ndvi_data['NDVI']
            ndvi_3d = np.expand_dims(ndvi_values, axis=2)  # Make it 3D
        else:
            # NDVI is regular array, ensure it's 3D
            if len(ndvi_data.shape) == 2:
                ndvi_3d = np.expand_dims(ndvi_data, axis=2)
            else:
                ndvi_3d = ndvi_data[:, :, :1]  # Take first band if 3D
        
        # For sensor data, ensure it's 3D
        if hasattr(sensor_data, 'dtype') and sensor_data.dtype.names is not None:
            # Sensor data is structured array, convert to regular 3D array
            # Get all band names
            band_names = sensor_data.dtype.names
            sensor_bands = []
            for band_name in band_names:
                band_data = sensor_data[band_name]
                if len(band_data.shape) == 2:
                    band_3d = np.expand_dims(band_data, axis=2)
                else:
                    band_3d = band_data
                sensor_bands.append(band_3d)
            # Concatenate all sensor bands along the band axis
            sensor_3d = np.concatenate(sensor_bands, axis=2)
        else:
            # Sensor data is regular array
            sensor_3d = sensor_data
        
        # Combine NDVI and sensor data along the band axis (axis=2)
        # Result should be (height, width, 21) - 1 NDVI + 20 sensor bands
        combined_array = np.concatenate([ndvi_3d, sensor_3d], axis=2)
        
        logger.info(f"Combined array shape: {combined_array.shape}")
        logger.info(f"✅ Successfully combined NDVI (1 band) + Sensor ({sensor_3d.shape[2]} bands) = {combined_array.shape[2]} bands total")
        
        # Import crop_and_pad function
        from modules.image_utils import crop_and_pad
        
        # Crop and pad to required shape (256x256)
        processed_array = crop_and_pad(combined_array, target_height=256, target_width=256)
        
        logger.info(f"Processed array shape after cropping and padding: {processed_array.shape}")
        
        return processed_array
        
    except Exception as e:
        logger.error(f"Error combining and processing data: {e}")
        return None

def save_rgb_image(data, output_path):
    """Save RGB image from data with proper Sentinel-2 scaling and contrast after cropping and padding."""
    try:
        # Handle both structured and regular arrays
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            # For structured arrays
            if 'Red' in data.dtype.names and 'Green' in data.dtype.names and 'Blue' in data.dtype.names:
                red = data['Red'].astype(np.float32)
                green = data['Green'].astype(np.float32)
                blue = data['Blue'].astype(np.float32)
            else:
                logger.error("Data does not contain expected RGB band names")
                return False
        else:
            # For regular arrays, ensure we have at least 3 bands
            if data.ndim >= 3 and data.shape[2] >= 3:
                red = data[:, :, 0].astype(np.float32)
                green = data[:, :, 1].astype(np.float32)
                blue = data[:, :, 2].astype(np.float32)
            else:
                logger.error("Not enough bands for RGB image")
                return False

        # Convert from DN (Digital Number) to Surface Reflectance (SR)
        # S2_SR data is scaled by 10000. Divide by 10000 to get [0.0, 1.0] reflectance.
        red_sr = red / 10000.0
        green_sr = green / 10000.0
        blue_sr = blue / 10000.0
        
        # Apply a contrast stretch for better visualization
        # Clip to remove outliers and normalize to 0-1 range with a reasonable max value
        VIS_MAX = 0.3  # Adjust this value to control brightness
        
        red_display = np.clip(red_sr, 0, VIS_MAX) / VIS_MAX
        green_display = np.clip(green_sr, 0, VIS_MAX) / VIS_MAX
        blue_display = np.clip(blue_sr, 0, VIS_MAX) / VIS_MAX

        # Stack the RGB bands for display
        rgb_image_data = np.stack([red_display, green_display, blue_display], axis=-1)
        
        # Ensure values are in the correct range [0, 1]
        rgb_image_data = np.clip(rgb_image_data, 0, 1)
        
        # Import crop_and_pad function
        from modules.image_utils import crop_and_pad
        
        # Crop and pad RGB image to 256x256
        if rgb_image_data.ndim == 2:
            # Grayscale image
            processed_rgb = crop_and_pad(rgb_image_data, target_height=256, target_width=256)
        else:
            # Color image - process each channel
            channels = []
            for i in range(rgb_image_data.shape[2]):
                channel_padded = crop_and_pad(rgb_image_data[:, :, i], target_height=256, target_width=256)
                channels.append(channel_padded)
            processed_rgb = np.stack(channels, axis=2)
        
        # Save the image
        plt.figure(figsize=(10, 8))
        plt.imshow(processed_rgb)
        plt.title("Sentinel-2 RGB Image", fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved RGB image to: {output_path}")
        return True
            
    except Exception as e:
        logger.error(f"Error saving RGB image: {e}")
        return False

def save_ndvi_image(data, output_path):
    """Save NDVI image with red to green colormap after cropping and padding."""
    try:
        # Handle both structured and regular arrays
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            # Structured array, extract NDVI band
            if 'NDVI' in data.dtype.names:
                data = data['NDVI']
            else:
                # Take first band if 3D structured array
                data = data[list(data.dtype.names)[0]]
        elif len(data.shape) > 2:
            # Regular 3D array, take first band
            data = data[:, :, 0]
        
        # Handle NaN and infinite values
        data = np.nan_to_num(data, nan=-1, posinf=-1, neginf=-1)
        
        # Clip NDVI values to valid range [-1, 1]
        data = np.clip(data, -1, 1)
        
        # Import crop_and_pad function
        from modules.image_utils import crop_and_pad
        
        # Crop and pad NDVI data to 128x128
        processed_ndvi = crop_and_pad(data, target_height=128, target_width=128)
        
        # Create red to green colormap (more distinct colors)
        colors = [
            (0.5, 0, 0),    # Dark Red (low values)
            (1, 1, 0),      # Yellow (medium values)
            (0, 0.5, 0)     # Dark Green (high values)
        ]
        red_to_green_cmap = LinearSegmentedColormap.from_list("RedToGreen", colors)
        
        # Create the visualization with black padding
        plt.figure(figsize=(12, 8))
        im = plt.imshow(processed_ndvi, cmap=red_to_green_cmap, origin='lower', vmin=-1, vmax=1, extent=[0, 128, 0, 128])
        
        # Set background to black
        plt.gca().set_facecolor('black')
        
        # Add colorbar with proper labels
        cbar = plt.colorbar(im, label='NDVI Values', ticks=[-1, -0.5, 0, 0.5, 1])
        cbar.ax.set_yticklabels(['-1 (Water)', '-0.5 (Bare Soil)', '0 (Sparse Veg.)', '0.5 (Moderate Veg.)', '1 (Dense Veg.)'])
        
        # Add statistics to title
        valid_data = processed_ndvi[processed_ndvi != -1]  # Exclude no-data values
        if len(valid_data) > 0:
            mean_ndvi = np.mean(valid_data)
            std_ndvi = np.std(valid_data)
            title = f"NDVI Heatmap (Mean: {mean_ndvi:.3f}, Std: {std_ndvi:.3f})"
        else:
            title = "NDVI Heatmap"
            
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.tight_layout()
        
        # Save the heatmap with black background
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        logger.info(f"Saved NDVI heatmap to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving NDVI image: {e}")
        return False

def search_satellite_image_with_source(polygon):
    """Search for satellite image and return both image and satellite name"""
    # For compatibility, just return the image from the main search function
    image = search_satellite_image(polygon, DATE_RANGE_START, DATE_RANGE_END, CLOUD_THRESHOLD)
    if image:
        return (image, 'Sentinel-2')
    return None

def select_rgb_bands_for_satellite(image, satellite_name):
    """Select RGB bands based on satellite source"""
    # For now, just use the standard Sentinel-2 approach
    return select_rgb_bands(image)

def search_satellite_image(polygon, start_date, end_date, cloud_threshold):
    """Search for satellite image in GEE for given polygon and date range"""
    try:
        logger.info(f"Searching for Sentinel-2 images from {start_date} to {end_date} with cloud threshold {cloud_threshold}%")
        
        # Search for Sentinel-2 imagery (using HARMONIZED version)
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        # Check collection size
        size = collection.size().getInfo()
        logger.info(f"Found {size} images in collection")
        
        if size == 0:
            logger.info("No images found with current filters, trying without cloud filter...")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(polygon)
                         .filterDate(start_date, end_date)
                         .sort('CLOUDY_PIXEL_PERCENTAGE'))
            size = collection.size().getInfo()
            logger.info(f"Found {size} images without cloud filter")
            
            if size == 0:
                # Try with extended date ranges (buffer periods)
                logger.info("No images found, trying with buffer periods...")
                
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
                    
                    logger.info(f"Trying extended date range: {extended_start_str} to {extended_end_str}")
                    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                 .filterBounds(polygon)
                                 .filterDate(extended_start_str, extended_end_str)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
                    
                    size = collection.size().getInfo()
                    logger.info(f"Found {size} images in extended date range")
                    
                    if size == 0:
                        # Try with no cloud filter in extended range
                        logger.info("No images found with cloud filter in extended range, trying without cloud filter...")
                        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                     .filterBounds(polygon)
                                     .filterDate(extended_start_str, extended_end_str)
                                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
                        size = collection.size().getInfo()
                        logger.info(f"Found {size} images in extended range without cloud filter")
                        
                        if size == 0:
                            logger.info("No images found even with extended date range and no cloud filter")
                            return None
                except Exception as date_error:
                    logger.error(f"Error with date extension: {date_error}")
                    return None
        
        # Get the first (least cloudy) image
        image = collection.first()
        
        # Check if image exists
        info = image.getInfo()
        if info and 'id' in info:
            logger.info(f"Selected image: {info['id']}")
            return image
        else:
            logger.info("No suitable image found")
            return None
    except Exception as e:
        logger.error(f"Error searching for satellite image: {e}")
        return None

def process_geojson_feature(area_name, feature, coord_index, feature_id=None):
    """Process a single GeoJSON feature"""
    try:
        # Use provided feature_id or generate one
        if feature_id is None:
            feature_id = f"feature_{coord_index}"
        
        logger.info(f"Processing feature {coord_index}: {feature_id}")
        
        # Extract coordinates
        coordinates = feature['geometry']['coordinates'][0]  # Exterior ring
        
        # Create Earth Engine polygon
        polygon = create_geometry_from_coordinates(coordinates)
        if polygon is None:
            logger.error(f"Failed to create polygon for feature {coord_index}")
            return False
        
        # Search for satellite image
        logger.info(f"Searching for satellite image for feature {coord_index}")
        image_result = search_satellite_image_with_source(polygon)
        
        if image_result is None:
            logger.warning(f"No suitable image found for feature {coord_index}")
            return False  # Skip this feature
        
        image, satellite_name = image_result
        
        # Select RGB bands based on satellite
        logger.info(f"Selecting RGB bands for {satellite_name}")
        rgb_image = select_rgb_bands_for_satellite(image, satellite_name)
        if rgb_image is None:
            logger.error(f"Failed to select RGB bands for feature {coord_index}")
            return False
        
        # Calculate NDVI
        logger.info(f"Calculating NDVI for {satellite_name}")
        ndvi_image = calculate_ndvi(image)
        if ndvi_image is None:
            logger.error(f"Failed to calculate NDVI for feature {coord_index}")
            return False
        
        # Get sensor data
        logger.info("Fetching sensor data")
        sensor_image = get_sensor_data(polygon)
        if sensor_image is None:
            logger.error(f"Failed to get sensor data for feature {coord_index}")
            return False
        
        # Export RGB data
        logger.info("Exporting RGB data")
        rgb_data = export_image_data(rgb_image, polygon, scale=10)
        if rgb_data is None:
            logger.error(f"Failed to export RGB data for feature {coord_index}")
            return False
        
        # Export NDVI data
        logger.info("Exporting NDVI data")
        ndvi_data = export_ndvi_data(ndvi_image, polygon, scale=10)
        if ndvi_data is None:
            logger.error(f"Failed to export NDVI data for feature {coord_index}")
            return False
        
        # Export sensor data
        logger.info("Exporting sensor data")
        sensor_data = export_sensor_data(sensor_image, polygon, scale=10)
        if sensor_data is None:
            logger.error(f"Failed to export sensor data for feature {coord_index}")
            return False
        
        # Combine NDVI and sensor data into 21-band array
        logger.info("Combining NDVI and sensor data into 21-band array")
        stacked_data = combine_ndvi_sensor_data(ndvi_data, sensor_data)
        if stacked_data is None:
            logger.error(f"Failed to combine data for feature {coord_index}")
            return False
        
        # Create output directory only after successful data retrieval
        output_dir = os.path.join(OUTPUT_BASE_DIR, area_name, str(coord_index))
        os.makedirs(output_dir, exist_ok=True)
        
        # Save stacked data (21-band numpy array)
        stacked_filename = f"{feature_id}_stacked.npy"
        stacked_path = os.path.join(output_dir, stacked_filename)
        np.save(stacked_path, stacked_data)
        logger.info(f"Saved stacked data to: {stacked_path}")
        
        # Save RGB image
        rgb_filename = f"{feature_id}_rgb.png"
        rgb_path = os.path.join(output_dir, rgb_filename)
        if save_rgb_image(rgb_data, rgb_path):
            logger.info(f"Saved RGB image to: {rgb_path}")
        else:
            logger.warning(f"Failed to save RGB image for feature {coord_index}")
        
        # Save NDVI image
        ndvi_filename = f"{feature_id}_ndvi.png"
        ndvi_path = os.path.join(output_dir, ndvi_filename)
        if save_ndvi_image(ndvi_data, ndvi_path):
            logger.info(f"Saved NDVI image to: {ndvi_path}")
        else:
            logger.warning(f"Failed to save NDVI image for feature {coord_index}")
        
        logger.info(f"✅ Successfully processed feature {coord_index} using {satellite_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing feature {coord_index}: {e}")
        return False

def process_geojson_areas(config_path):
    """Process all areas in the GeoJSON configuration"""
    # Load configuration
    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # Get area name
    area_name = config.get('area', {}).get('name', 'unknown_area')
    logger.info(f"Processing area: {area_name}")
    
    # Get features from geojson
    geojson_data = config.get('area', {}).get('geojson', {})
    features = geojson_data.get('features', [])
    logger.info(f"Found {len(features)} features in geojson")
    
    # Process each feature
    processed_count = 0
    skipped_count = 0
    
    for i, feature in enumerate(features):
        coord_index = i + 1
        feature_id = feature.get('properties', {}).get('id', f"feature_{coord_index}")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing coordinate {coord_index}/{len(features)}")
        logger.info(f"{'='*50}")
        
        try:
            success = process_geojson_feature(area_name, feature, coord_index, feature_id)
            if success:
                processed_count += 1
            else:
                skipped_count += 1
                logger.warning(f"Skipped feature {coord_index} due to data issues")
            
        except Exception as e:
            skipped_count += 1
            logger.error(f"Error processing feature {coord_index}: {e}")
            continue
    
    logger.info(f"\n{'='*50}")
    logger.info("Processing complete")
    logger.info(f"Processed: {processed_count} features")
    logger.info(f"Skipped: {skipped_count} features")
    logger.info(f"{'='*50}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process GeoJSON areas with satellite and sensor data')
    parser.add_argument('--config', default='./config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        logger.error("Failed to initialize Earth Engine. Exiting.")
        return
    
    # Process geojson areas
    process_geojson_areas(args.config)

if __name__ == "__main__":
    main()