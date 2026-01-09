import numpy as np
import rasterio

def compute_indices_from_image(img_array):
    """
    Compute multiple vegetation/water/nutrient indices and aggregate statistics.
    Returns a dictionary of means, stds, mins, maxs for each index.
    """
    if img_array.shape[0] < 6:
        raise ValueError("Expected at least 6 bands")

    B, G, R, NIR, SWIR1, SWIR2 = img_array[:6]
    eps = 1e-6

    indices = {}
    indices['NDVI']    = (NIR - R) / (NIR + R + eps)
    indices['EVI']     = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1 + eps)
    indices['SAVI']    = ((NIR - R) / (NIR + R + 0.5 + eps)) * 1.5
    indices['GNDVI']   = (NIR - G) / (NIR + G + eps)
    indices['CIgreen'] = (NIR / (G + eps)) - 1
    indices['MCARI']   = ((NIR - R) - 0.2*(NIR - G)) * (NIR / (R + eps))
    indices['NDWI']    = (G - NIR) / (G + NIR + eps)
    indices['MSI']     = SWIR1 / (NIR + eps)
    indices['ARI']     = (1 / (G + eps)) - (1 / (R + eps))
    indices['SIPI']    = (NIR - B) / (NIR - R + eps)
    indices['NDBI']    = (SWIR1 - NIR) / (SWIR1 + NIR + eps)

    aggregated = {}
    for name, arr in indices.items():
        aggregated[f"{name}_mean"] = float(np.nanmean(arr))
        aggregated[f"{name}_std"]  = float(np.nanstd(arr))
        aggregated[f"{name}_min"]  = float(np.nanmin(arr))
        aggregated[f"{name}_max"]  = float(np.nanmax(arr))

    return aggregated

def resize_image(img_array, target=(64,64)):
    import cv2
    # convert HWC to CHW if needed
    if img_array.shape[2] <= 6:
        img_array = np.transpose(img_array, (2,0,1))
    bands, h, w = img_array.shape
    resized = [cv2.resize(img_array[b], target, interpolation=cv2.INTER_LINEAR) for b in range(bands)]
    return np.stack(resized)
