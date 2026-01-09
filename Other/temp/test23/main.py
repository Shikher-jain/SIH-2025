"""
Main script to process coordinates and fetch sensor data from Google Earth Engine.
Downloads N, OC, ECE, P, pH sensor data for specified coordinates.
"""
 
import ee
import numpy as np
import json
import os
from datetime import datetime
import argparse

def initialize_earth_engine(service_account_path="./earth-engine-service-account.json"):
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

def create_point_geometry(coords):
    """Create GEE point geometry from coordinates"""
    try:
        return ee.Geometry.Point(coords)
    except Exception as e:
        print(f"Error creating point geometry: {e}")
        return None

def create_rectangle_geometry(coords):
    """Create GEE rectangle geometry from bounding box coordinates"""
    try:
        # Assuming coords is a list of [min_lon, min_lat, max_lon, max_lat]
        if len(coords) == 4:
            return ee.Geometry.Rectangle(coords)
        else:
            print("Invalid rectangle coordinates format. Expected [min_lon, min_lat, max_lon, max_lat]")
            return None
    except Exception as e:
        print(f"Error creating rectangle geometry: {e}")
        return None

def create_polygon_geometry(coords):
    """Create GEE polygon geometry from polygon coordinates"""
    try:
        # Assuming coords is a list of [[lon1, lat1], [lon2, lat2], ...]
        geometry = ee.Geometry.Polygon([coords])
        print(f"      Created polygon with {len(coords)} vertices")
        return geometry
    except Exception as e:
        print(f"Error creating polygon geometry: {e}")
        return None

def get_sensor_data_for_point(point, date):
    """Fetch sensor data for a specific point and date"""
    try:
        # Sensor asset IDs
        sensor_assets = {
            'ECe': 'projects/sih2k25-472714/assets/ECe',
            'N': 'projects/sih2k25-472714/assets/N',
            'P': 'projects/sih2k25-472714/assets/P',
            'OC': 'projects/sih2k25-472714/assets/OC',
            'pH': 'projects/sih2k25-472714/assets/pH'
        }
        
        # Dictionary to store sensor data
        sensor_data = {}
        
        # Fetch data for each sensor
        for sensor_name, asset_id in sensor_assets.items():
            try:
                print(f"  Fetching {sensor_name} data...")
                sensor_image = ee.Image(asset_id)
                
                # Sample the image at the point
                point_data = sensor_image.sample(
                    region=point,
                    scale=10,  # 10m resolution
                    numPixels=1,
                    date=date
                ).first()
                
                # Get the data
                info = point_data.getInfo()
                if info and 'properties' in info:
                    # Get all properties (should be 4 bands for each sensor)
                    properties = info['properties']
                    sensor_data[sensor_name] = properties
                    print(f"    ✅ {sensor_name}: {properties}")
                else:
                    print(f"    ❌ No data found for {sensor_name}")
                    sensor_data[sensor_name] = None
                    
            except Exception as e:
                print(f"    ❌ Error fetching {sensor_name} data: {e}")
                sensor_data[sensor_name] = None
        
        return sensor_data
        
    except Exception as e:
        print(f"Error fetching sensor data: {e}")
        return None

def export_sensor_data(sensor_image, region, scale=10):
    """Export sensor data as numpy array with specified scale (meters per pixel)"""
    try:
        print(f"Exporting sensor data with scale {scale} meters per pixel...")
        
        # Clip the image to the region
        clipped = sensor_image.clip(region)
        
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
        
        # Calculate dimensions in pixels (using 315x316 as in test15 config)
        width = 316
        height = 315
        
        print(f"Using fixed image dimensions: {height}x{width} pixels")
        
        # Define grid parameters based on scale
        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height
        
        # Create request for computePixels
        request = {
            'expression': clipped,
            'fileFormat': 'NUMPY_NDARRAY',
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
        print("Fetching sensor pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)
        
        if pixel_data is not None:
            print(f"Successfully fetched sensor data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            print("Failed to fetch sensor data")
            return None
            
    except Exception as e:
        print(f"Error exporting sensor data: {e}")
        return None

def get_sensor_data_for_region(region, date):
    """Fetch sensor data for a specific region and date - 1 band per sensor"""
    try:
        # Sensor asset IDs
        sensor_assets = {
            'ECe': 'projects/sih2k25-472714/assets/ECe',
            'N': 'projects/sih2k25-472714/assets/N',
            'P': 'projects/sih2k25-472714/assets/P',
            'OC': 'projects/sih2k25-472714/assets/OC',
            'pH': 'projects/sih2k25-472714/assets/pH'
        }
        
        # List to store one band per sensor
        sensor_bands = []
        
        # Fetch data for each sensor (only 1 band per sensor)
        for sensor_name, asset_id in sensor_assets.items():
            try:
                print(f"  Fetching {sensor_name} data...")
                sensor_image = ee.Image(asset_id)
                
                # Get information about the image to see available bands
                try:
                    image_info = sensor_image.getInfo()
                    if 'bands' in image_info:
                        band_names = [band['id'] for band in image_info['bands']]
                        print(f"    Available bands: {band_names}")
                except Exception as info_error:
                    print(f"    Could not get band info: {info_error}")
                
                # Select only the first band (b1) for each sensor, which is typically the mean
                sensor_band = sensor_image.select('b1')  # Select first band (mean values)
                
                # Export the data with 10m resolution
                sensor_data = export_sensor_data(sensor_band, region, scale=10)
                
                if sensor_data is not None:
                    print(f"    ✅ {sensor_name}: shape {sensor_data.shape}")
                    # If it's a 3D array, take the first band
                    if len(sensor_data.shape) == 3:
                        sensor_bands.append(sensor_data[:, :, 0])
                    else:
                        # If it's 2D, use as is
                        sensor_bands.append(sensor_data)
                else:
                    print(f"    ❌ No data found for {sensor_name}")
                    # Add empty band as placeholder
                    sensor_bands.append(np.zeros((315, 316)))
                    
            except Exception as e:
                print(f"    ❌ Error fetching {sensor_name} data: {e}")
                # Add empty band as placeholder
                sensor_bands.append(np.zeros((315, 316)))
        
        # Stack all bands into a 3D array (315, 316, 5)
        if sensor_bands:
            sensor_stack = np.stack(sensor_bands, axis=2)
            print(f"  ✅ Combined sensor data shape: {sensor_stack.shape}")
            return sensor_stack
        else:
            print(f"  ❌ Failed to fetch any sensor data")
            return np.zeros((315, 316, 5))
        
    except Exception as e:
        print(f"Error fetching sensor data: {e}")
        return np.zeros((315, 316, 5))

def save_sensor_data_as_npy(sensor_data, output_dir, district_number):
    """Save sensor data as .npy file in 3D array format (315, 316, 5)"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = f"{district_number}.npy"
        filepath = os.path.join(output_dir, filename)
        
        # Save data - should be a 3D array with shape (height, width, 5)
        np.save(filepath, sensor_data)
        print(f"  ✅ Saved sensor data to: {filepath}")
        print(f"  📊 Data shape: {sensor_data.shape}")
        
        return filepath
    except Exception as e:
        print(f"  ❌ Error saving sensor data: {e}")
        return None

def process_coordinates(coords_data, date, output_dir):
    """Process all coordinates and fetch sensor data"""
    print(f"Processing coordinates for date: {date}")
    print(f"Output directory: {output_dir}")
    
    # Counter for district numbering
    district_counter = 1
    
    # Process each state
    for state_name, districts in coords_data.items():
        print(f"\n📍 Processing state: {state_name}")
        
        # Process each district
        for district_name, coordinates_list in districts.items():
            print(f"  🏘️ Processing district: {district_name}")
            
            # Process each coordinate set
            for i, coords in enumerate(coordinates_list):
                print(f"    📍 Processing coordinate set {i+1}/{len(coordinates_list)}")
                
                # Determine geometry type based on coordinate structure
                geometry = None
                if isinstance(coords, list) and len(coords) >= 3:
                    # Polygon coordinates [[lon1, lat1], [lon2, lat2], ...]
                    # Check if it's a nested list (your format) or flat list
                    if isinstance(coords[0], list) and len(coords[0]) == 2:
                        # Nested format: [[lon1, lat1], [lon2, lat2], ...]
                        geometry = create_polygon_geometry(coords)
                        print(f"      Creating polygon geometry with {len(coords)} vertices")
                    else:
                        print(f"      ❌ Invalid coordinate format: {coords}")
                        continue
                else:
                    print(f"      ❌ Invalid coordinate format: {coords}")
                    continue
                
                if geometry is None:
                    print(f"      ❌ Failed to create geometry")
                    continue
                
                # Fetch sensor data
                print(f"    📡 Fetching sensor data...")
                sensor_data = get_sensor_data_for_region(geometry, date)
                
                if sensor_data is None:
                    print(f"      ❌ Failed to fetch sensor data")
                    continue
                
                if isinstance(date, str):
                    date_obj = datetime.strptime(date, "%Y-%m-%d")  # ya format jaisa tumhara ho
                    year = date_obj.year
                else:
                    year = date.year
                # Save sensor data with new naming convention: {district_name}_{district_number}
                filename = f"{district_name}_{district_counter}_{year}_Sensor"
                print(f"    💾 Saving data as {filename}.npy")
                save_sensor_data_as_npy(sensor_data, output_dir, filename)
                
                # Increment district counter
                district_counter += 1
    
    print(f"\n🎉 Processing complete! Processed {district_counter-1} coordinate sets.")

def load_config(config_path="./config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def main():
    print("GEE Sensor Data Downloader")
    print("=" * 30)
    
    # Load configuration
    config = load_config("./config.json")
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    
    # Get coordinates data from config
    coords_data = config.get("coordinates", {})
    
    # Get date from config or command line
    parser = argparse.ArgumentParser(description='Download sensor data from Google Earth Engine')
    parser.add_argument('--date', type=str, help='Date for data retrieval (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Use date from command line if provided, otherwise from config
    date = args.date if args.date else config.get("date", "2023-03-15")
    
    # Use output directory from command line if provided, otherwise from config
    output_dir = args.output if args.output else config.get("output_dir", "./output")
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        print("Failed to initialize Earth Engine. Exiting.")
        return
    
    # Process coordinates
    process_coordinates(coords_data, date, output_dir)

if __name__ == "__main__":
    main()