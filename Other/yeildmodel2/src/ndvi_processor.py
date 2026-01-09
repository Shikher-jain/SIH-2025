import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def calculate_ndvi(nir_band, red_band):
    """Calculate NDVI from NIR and Red bands."""
    try:
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        return ndvi
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        return None

def generate_ndvi_heatmap(ndvi_data, threshold_low=0.3, threshold_high=0.6):
    """Generate a red-yellow-green mask based on NDVI values."""
    try:
        # Normalize NDVI values to [0, 1]
        normalized_ndvi = (ndvi_data - np.nanmin(ndvi_data)) / (np.nanmax(ndvi_data) - np.nanmin(ndvi_data))
        normalized_ndvi = np.nan_to_num(normalized_ndvi, nan=0.0)

        # Create a colormap: red (poor) to yellow (moderate) to green (healthy)
        colors = [
            (1, 0, 0),      # Red
            (1, 1, 0),      # Yellow
            (0, 1, 0)       # Green
        ]
        cmap = LinearSegmentedColormap.from_list("NDVI_Colors", colors)

        # Create mask based on thresholds
        mask = np.zeros_like(normalized_ndvi)
        mask[normalized_ndvi < threshold_low] = 0  # Poor
        mask[(normalized_ndvi >= threshold_low) & (normalized_ndvi < threshold_high)] = 1  # Moderate
        mask[normalized_ndvi >= threshold_high] = 2  # Healthy

        # Create an RGBA image
        rgba_image = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba_image[..., 3] = 1  # Set alpha channel to fully opaque

        # Assign colors based on mask
        rgba_image[mask == 0] = [1, 0, 0, 1]  # Red
        rgba_image[mask == 1] = [1, 1, 0, 1]  # Yellow
        rgba_image[mask == 2] = [0, 1, 0, 1]  # Green

        return rgba_image
    except Exception as e:
        print(f"Error generating NDVI heatmap: {e}")
        return None

def process_ndvi_image(nir_band, red_band):
    """Process NDVI image from NIR and Red bands."""
    try:
        ndvi = calculate_ndvi(nir_band, red_band)
        if ndvi is not None:
            heatmap = generate_ndvi_heatmap(ndvi)
            return heatmap
        else:
            print("NDVI calculation failed.")
            return None
    except Exception as e:
        print(f"Error processing NDVI image: {e}")
        return None