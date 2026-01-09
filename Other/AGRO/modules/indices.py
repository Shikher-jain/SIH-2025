# modules/indices.py
import numpy as np
import rasterio

def compute_indices_from_image(sat_file):
    """
    Compute multiple indices (NDVI, EVI, SAVI, GNDVI, NDWI, etc.)
    Returns aggregated statistics: mean, std, min, max
    """
    with rasterio.open(sat_file) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)

    if arr.shape[0] < 6:
        raise ValueError(f"Expected 6 bands, got {arr.shape[0]}")

    B, G, R, NIR, SWIR1, SWIR2 = arr[:6]
    eps = 1e-6

    indices = {}
    indices['NDVI']    = (NIR - R) / (NIR + R + eps)
    indices['EVI']     = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1 + eps)
    indices['SAVI']    = ((NIR - R) / (NIR + R + 0.5 + eps)) * 1.5
    indices['GNDVI']   = (NIR - G) / (NIR + G + eps)
    indices['CIgreen'] = (NIR / (G + eps)) - 1
    indices['MCARI']   = ((NIR - R) - 0.2*(NIR - G)) * (NIR / (R + eps))
    indices['NDWI'] = (G - NIR) / (G + NIR + eps)
    indices['MSI']  = SWIR1 / (NIR + eps)
    indices['ARI']  = (1 / (G + eps)) - (1 / (R + eps))
    indices['SIPI'] = (NIR - B) / (NIR - R + eps)
    indices['NDBI'] = (SWIR1 - NIR) / (SWIR1 + NIR + eps)

    aggregated = {}
    for name, arr in indices.items():
        aggregated[f"{name}_mean"] = float(np.nanmean(arr))
        aggregated[f"{name}_std"]  = float(np.nanstd(arr))
        aggregated[f"{name}_min"]  = float(np.nanmin(arr))
        aggregated[f"{name}_max"]  = float(np.nanmax(arr))

    return aggregated
def compute_indices_from_image(img_array):
    """
    img_array: numpy array (bands, H, W)
    Sentinel-2 band example: B2-Green, B3-Red, B4-NIR, B5-SWIR
    """
    if img_array.shape[0] < 6:
        raise ValueError("Expected at least 6 bands")
    
    red   = img_array[3]
    green = img_array[2]
    nir   = img_array[7] if img_array.shape[0] > 7 else img_array[3]
    swir  = img_array[5]

    ndvi  = (nir - red) / (nir + red + 1e-8)
    gndvi = (nir - green) / (nir + green + 1e-8)
    ndwi  = (green - nir) / (green + nir + 1e-8)

    indices = {
        "NDVI": ndvi.tolist(),
        "GNDVI": gndvi.tolist(),
        "NDWI": ndwi.tolist()
    }

    # Aggregated stats
    for k, arr in indices.items():
        arr_np = np.array(arr, dtype=np.float32)
        indices[f"{k}_mean"] = float(np.mean(arr_np))
        indices[f"{k}_std"] = float(np.std(arr_np))
        indices[f"{k}_min"] = float(np.min(arr_np))
        indices[f"{k}_max"] = float(np.max(arr_np))

    return indices


def resize_image(img_array, target=(64,64)):
    import cv2
    # convert HWC -> CHW if needed
    if img_array.shape[2] <= 6:
        img_array = np.transpose(img_array, (2,0,1))
    bands, h, w = img_array.shape
    resized = [cv2.resize(img_array[b], target, interpolation=cv2.INTER_LINEAR) for b in range(bands)]
    return np.stack(resized)

