import numpy as np
import matplotlib.pyplot as plt

def generate_mask(yield_data, thresholds=(0.3, 0.6)):
    """Generate a red-yellow-green mask based on yield values."""
    if yield_data.ndim != 2:
        raise ValueError("Yield data must be a 2D numpy array.")

    # Create an empty mask with the same shape as yield_data
    mask = np.zeros((*yield_data.shape, 4), dtype=np.uint8)  # RGBA

    # Apply thresholds to create the mask
    mask[..., 3] = 255  # Set alpha channel to fully opaque

    # Red for low yield
    low_mask = yield_data <= thresholds[0]
    mask[low_mask] = [255, 0, 0, 255]  # Red

    # Yellow for moderate yield
    moderate_mask = (yield_data > thresholds[0]) & (yield_data <= thresholds[1])
    mask[moderate_mask] = [255, 255, 0, 255]  # Yellow

    # Green for high yield
    high_mask = yield_data > thresholds[1]
    mask[high_mask] = [0, 255, 0, 255]  # Green

    return mask

def process_ndvi_data(ndvi_data):
    """Process NDVI data and return a mask."""
    if ndvi_data.ndim != 2:
        raise ValueError("NDVI data must be a 2D numpy array.")

    # Normalize NDVI data to range [0, 1]
    normalized_ndvi = (ndvi_data - np.nanmin(ndvi_data)) / (np.nanmax(ndvi_data) - np.nanmin(ndvi_data))
    normalized_ndvi = np.nan_to_num(normalized_ndvi)

    # Generate a mask based on NDVI thresholds
    return generate_mask(normalized_ndvi)