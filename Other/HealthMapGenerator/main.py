"""
Main script to process GeoJSON coordinates from config.json, fetch satellite images through GEE,
create heatmaps, and store them in the required folder structure.
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

def initialize_earth_engine(service_account_path="earth-engine-service-account.json"):
    """Initialize Google Earth Engine with service account authentication"""
    try:
        # Load service account credentials
        with open(service_account_path, 'r') as f:
            service_account = json.load(f)
        
        print(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")
        
        # Authenticate and initialize
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

def create_geometry_from_geojson(geojson_feature):
    """Create GEE geometry from GeoJSON feature"""
    try:
        coordinates = geojson_feature['geometry']['coordinates']
        if geojson_feature['geometry']['type'] == 'Polygon':
            return ee.Geometry.Polygon(coordinates)
        elif geojson_feature['geometry']['type'] == 'MultiPolygon':
            return ee.Geometry.MultiPolygon(coordinates)
        else:
            raise ValueError(f"Unsupported geometry type: {geojson_feature['geometry']['type']}")
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
                    from dateutil.relativedelta import relativedelta
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

def select_rgb_bands(image):
    """Select RGB bands (B4, B3, B2) from Sentinel-2 image"""
    try:
        # Select RGB bands: B4 (Red), B3 (Green), B2 (Blue)
        rgb_image = image.select(['B4', 'B3', 'B2']).rename(['Red', 'Green', 'Blue'])
        return rgb_image
    except Exception as e:
        print(f"Error selecting RGB bands: {e}")
        return None

def select_ndvi_bands(image):
    """Select NDVI bands (B8-NIR, B4-Red) from Sentinel-2 image for crop health"""
    try:
        # Select NIR and Red bands for NDVI calculation
        ndvi_bands = image.select(['B8', 'B4']).rename(['NIR', 'Red'])
        return ndvi_bands
    except Exception as e:
        print(f"Error selecting NDVI bands: {e}")
        return None

def calculate_ndvi(image):
    """Calculate NDVI from NIR and Red bands"""
    try:
        nir = image.select('NIR')
        red = image.select('Red')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return ndvi
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        return None

def export_image_data(image, region, scale=10, band_names=None):
    """Export image data as numpy array with specified scale (meters per pixel)"""
    try:
        print(f"Exporting image data with scale {scale} meters per pixel...")
        
        # Clip the image to the region
        clipped = image.clip(region)
        
        # Get band names from the image if not provided
        if band_names is None:
            band_info = image.getInfo()
            if 'bands' in band_info:
                band_names = [band['id'] for band in band_info['bands']]
            else:
                band_names = ['band']
        
        print(f"Exporting bands: {band_names}")
        
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
        
        print(f"Calculated image dimensions: {width}x{height} pixels")
        
        # Define grid parameters based on scale
        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height
        
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
        print("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)
        
        if pixel_data is not None:
            print(f"Successfully fetched pixel data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            print("Failed to fetch pixel data")
            return None
            
    except Exception as e:
        print(f"Error exporting image data: {e}")
        return None

def crop_to_256(data):
    """Crop data to 256x256 pixels, centered if possible"""
    try:
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            # For structured arrays, we need to crop each field
            if len(data.dtype.names) > 0:
                first_field = data.dtype.names[0]
                height, width = data[first_field].shape
                # Calculate crop coordinates (centered crop)
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)
                
                # Create new structured array with cropped data
                cropped_data = np.empty((end_y - start_y, end_x - start_x), dtype=data.dtype)
                for field_name in data.dtype.names:
                    cropped_data[field_name] = data[field_name][start_y:end_y, start_x:end_x]
                return cropped_data
        else:
            # For regular arrays
            if data.ndim >= 2:
                height, width = data.shape[:2]
                # Calculate crop coordinates (centered crop)
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)
                
                if data.ndim == 2:
                    return data[start_y:end_y, start_x:end_x]
                else:
                    return data[start_y:end_y, start_x:end_x, :]
        return data
    except Exception as e:
        print(f"Error cropping data to 256x256: {e}")
        return data

def crop_to_256(data):
    """Crop data to 256x256 pixels, centered if possible"""
    try:
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            # For structured arrays, we need to crop each field
            if 'Red' in data.dtype.names and 'Green' in data.dtype.names and 'Blue' in data.dtype.names:
                height, width = data['Red'].shape
                # Calculate crop coordinates (centered crop)
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)
                
                # Create new structured array with cropped data
                cropped_data = np.empty((end_y - start_y, end_x - start_x), dtype=data.dtype)
                cropped_data['Red'] = data['Red'][start_y:end_y, start_x:end_x]
                cropped_data['Green'] = data['Green'][start_y:end_y, start_x:end_x]
                cropped_data['Blue'] = data['Blue'][start_y:end_y, start_x:end_x]
                return cropped_data
        else:
            # For regular arrays
            if data.ndim >= 2:
                height, width = data.shape[:2]
                # Calculate crop coordinates (centered crop)
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)
                
                if data.ndim == 2:
                    return data[start_y:end_y, start_x:end_x]
                else:
                    return data[start_y:end_y, start_x:end_x, :]
        return data
    except Exception as e:
        print(f"Error cropping data: {e}")
        return data

def save_rgb_image(data, output_path):
    """Save RGB image from data with proper Sentinel-2 scaling and contrast."""
    try:
        if not (hasattr(data, 'dtype') and data.dtype.names is not None):
            print("Invalid data format for structured array.")
            return False

        # 1. Extract bands from structured numpy array
        if 'Red' not in data.dtype.names or 'Green' not in data.dtype.names or 'Blue' not in data.dtype.names:
             print("Data does not contain expected RGB band names ('Red', 'Green', 'Blue')")
             return False

        red = data['Red'].astype(np.float32)
        green = data['Green'].astype(np.float32)
        blue = data['Blue'].astype(np.float32)

        # 2. Convert from DN (Digital Number) to Surface Reflectance (SR)
        # S2_SR data is scaled by 10000. Divide by 10000 to get [0.0, 1.0] reflectance.
        red_sr = red / 10000.0
        green_sr = green / 10000.0
        blue_sr = blue / 10000.0
        
        # 3. Apply a contrast stretch (Normalization)
        # Typical natural scenes have max reflectance well below 1.0.
        # Clipping/stretching is essential for visual aesthetics.
        # A common, simple stretch is to clip to a max value like 0.3 or 0.4
        # and then normalize to [0, 1].
        
        # Use a maximum SR value for clipping (e.g., 0.3 for a standard stretch)
        VIS_MAX = 0.3
        
        red_display = np.clip(red_sr, 0, VIS_MAX) / VIS_MAX
        green_display = np.clip(green_sr, 0, VIS_MAX) / VIS_MAX
        blue_display = np.clip(blue_sr, 0, VIS_MAX) / VIS_MAX

        # 4. Stack the RGB bands for display
        rgb_image_data = np.stack([red_display, green_display, blue_display], axis=-1)
        
        # 5. Save/Display
        plt.figure(figsize=(10, 8))
        # Matplotlib's imshow expects float data in the range [0.0, 1.0]
        plt.imshow(rgb_image_data)
        plt.title("Sentinel-2 RGB Image (Stretched)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RGB image to: {output_path}")
        return True
            
    except Exception as e:
        print(f"Error saving RGB image with proper scaling: {e}")
        return False

def save_ndvi_heatmap(ndvi_data, output_dir, coord_number, area_name):
    """Save NDVI data as crop health heatmap"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Crop NDVI data to 256x256
        cropped_ndvi = crop_to_256(ndvi_data)
        
        # Extract NDVI values
        if hasattr(cropped_ndvi, 'dtype') and cropped_ndvi.dtype.names is not None:
            if 'NDVI' in cropped_ndvi.dtype.names:
                ndvi_values = cropped_ndvi['NDVI']
            else:
                # Use first available field
                field_name = cropped_ndvi.dtype.names[0]
                ndvi_values = cropped_ndvi[field_name]
        else:
            ndvi_values = cropped_ndvi
        
        # Ensure proper data type
        if ndvi_values.dtype != np.float32 and ndvi_values.dtype != np.float256:
            ndvi_values = ndvi_values.astype(np.float32)
        
        # NDVI values should be between -1 and 1, but let's normalize what we have
        data_min = np.nanmin(ndvi_values)
        data_max = np.nanmax(ndvi_values)
        
        if np.isnan(data_min) or np.isnan(data_max) or data_max == data_min:
            normalized_ndvi = np.zeros_like(ndvi_values)
        else:
            # Normalize to 0-1 range for visualization
            normalized_ndvi = (ndvi_values - data_min) / (data_max - data_min)
            normalized_ndvi = np.nan_to_num(normalized_ndvi, nan=0.0)
        
        # Create crop health colormap (red=unhealthy, yellow=moderate, green=healthy)
        colors = [
            (0.8, 0, 0),      # Dark red (very unhealthy)
            (1, 0.3, 0),      # Red-orange (unhealthy)
            (1, 1, 0),        # Yellow (moderate health)
            (0.5, 1, 0),      # Yellow-green (good health)
            (0, 0.8, 0)       # Dark green (very healthy)
        ]
        crop_health_cmap = LinearSegmentedColormap.from_list("CropHealth", colors)
        
        # Generate NDVI crop health heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(normalized_ndvi, cmap=crop_health_cmap, origin='lower')
        
        # Add colorbar with crop health interpretation
        cbar = plt.colorbar(im, label='Crop Health (Red=Poor, Green=Healthy)', ticks=[0, 0.25, 0.5, 0.75, 1])
        cbar.ax.set_yticklabels([
            f'Poor (Min: {data_min:.3f})', 
            'Stressed', 
            'Moderate', 
            'Good', 
            f'Healthy (Max: {data_max:.3f})'
        ])
        
        # Add labels and title
        plt.title(f"Crop Health (NDVI) Heatmap (256x256) - {area_name} Cell {coord_number}")
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.tight_layout()
        
        # Save the NDVI heatmap with new naming format
        ndvi_png_filename = f"{area_name}_{coord_number}_ndvi_heatmap.png"
        ndvi_png_path = os.path.join(output_dir, ndvi_png_filename)
        plt.savefig(ndvi_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved NDVI crop health heatmap to: {ndvi_png_path}")
        
        # Print NDVI statistics for crop health assessment
        valid_ndvi = ndvi_values[~np.isnan(ndvi_values)]
        if len(valid_ndvi) > 0:
            print(f"NDVI Statistics - Min: {np.min(valid_ndvi):.3f}, Max: {np.max(valid_ndvi):.3f}, Mean: {np.mean(valid_ndvi):.3f}")
            
            # Crop health interpretation
            healthy_pixels = np.sum(valid_ndvi > 0.6) if data_max > 0.6 else 0
            moderate_pixels = np.sum((valid_ndvi > 0.3) & (valid_ndvi <= 0.6)) if data_max > 0.3 else 0
            poor_pixels = np.sum(valid_ndvi <= 0.3)
            total_pixels = len(valid_ndvi)
            
            print(f"Crop Health Assessment:")
            print(f"  - Healthy vegetation: {healthy_pixels}/{total_pixels} pixels ({100*healthy_pixels/total_pixels:.1f}%)")
            print(f"  - Moderate vegetation: {moderate_pixels}/{total_pixels} pixels ({100*moderate_pixels/total_pixels:.1f}%)")
            print(f"  - Poor/Stressed vegetation: {poor_pixels}/{total_pixels} pixels ({100*poor_pixels/total_pixels:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error saving NDVI heatmap: {e}")
        return False
def process_geojson_coordinates(config):
    """Process each coordinate in geojson and generate NDVI heatmaps only"""
    # Get area name from config
    area_name = config.get('area', {}).get('name', 'unknown_area')
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Date range for satellite search (Oct 2017 to Mar 2018 with buffer)
    start_date = '2018-02-01'
    end_date = '2018-04-30'
    
    # Get features from geojson
    geojson_data = config.get('area', {}).get('geojson', {})
    features = geojson_data.get('features', [])
    print(f"Found {len(features)} coordinates in geojson")
    
    # Process each coordinate
    for i, feature in enumerate(features):
        coord_number = i + 1
        print(f"\n{'='*50}")
        print(f"Processing coordinate {coord_number}/{len(features)}")
        print(f"{'='*50}")
        
        try:
            # Create Earth Engine polygon
            polygon = create_geometry_from_geojson(feature)
            if polygon is None:
                print(f"Failed to create polygon for coordinate {coord_number}")
                continue
            
            # Search for satellite image
            print(f"Searching for satellite image between {start_date} and {end_date}")
            image = search_satellite_image(polygon, start_date, end_date)
            
            if image is None:
                print(f"No suitable image found for coordinate {coord_number}")
                continue
            
            # Calculate NDVI for crop health analysis (only output)
            print("Calculating NDVI for crop health analysis")
            ndvi_bands = select_ndvi_bands(image)
            if ndvi_bands is not None:
                ndvi_image = calculate_ndvi(ndvi_bands)
                if ndvi_image is not None:
                    print("Exporting NDVI data for crop health analysis")
                    ndvi_data = export_image_data(ndvi_image, polygon, scale=10, band_names=['NDVI'])
                    if ndvi_data is not None:
                        # Save NDVI heatmap (crop health) - only output we want
                        print(f"Generating NDVI crop health heatmap with filename: {area_name}_{coord_number}_ndvi_heatmap.png")
                        if save_ndvi_heatmap(ndvi_data, output_dir, coord_number, area_name):
                            print(f"✅ Successfully processed coordinate {coord_number}")
                        else:
                            print(f"❌ Failed to save NDVI data for coordinate {coord_number}")
                    else:
                        print(f"Failed to export NDVI data for coordinate {coord_number}")
                else:
                    print(f"Failed to calculate NDVI for coordinate {coord_number}")
            else:
                print(f"Failed to select NDVI bands for coordinate {coord_number}")
            
        except Exception as e:
            print(f"Error processing coordinate {coord_number}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Processing complete")
    print(f"{'='*50}")

def main():
    print("GeoJSON-based NDVI Crop Health Analysis")
    print("="*50)
    
    # Load configuration
    config_path = "./config.json"
    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    
    print("Configuration loaded successfully")
    
    # Get area name
    area_name = config.get('area', {}).get('name', 'unknown_area')
    print(f"Processing area: {area_name}")
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        print("Failed to initialize Earth Engine. Exiting.")
        return
    
    # Process geojson coordinates
    process_geojson_coordinates(config)

if __name__ == "__main__":
    main()