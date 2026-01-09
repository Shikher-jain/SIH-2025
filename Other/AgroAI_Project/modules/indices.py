import numpy as np
import rasterio

def compute_indices_from_image(sat_file):
    with rasterio.open(sat_file) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] < 6:
        raise ValueError("Expected 6 bands")
    B, G, R, NIR, SWIR1, SWIR2 = arr[:6]
    eps = 1e-6
    indices = {}
    indices['NDVI']  = (NIR - R) / (NIR + R + eps)
    indices['EVI']   = 2.5*(NIR - R)/(NIR + 6*R - 7.5*B + 1 + eps)
    indices['SAVI']  = ((NIR - R)/(NIR + R + 0.5 + eps))*1.5
    indices['GNDVI'] = (NIR - G)/(NIR + G + eps)
    indices['NDWI']  = (G - NIR)/(G + NIR + eps)
    agg = {}
    for k, arrv in indices.items():
        agg[f"{k}_mean"] = float(np.nanmean(arrv))
        agg[f"{k}_std"]  = float(np.nanstd(arrv))
        agg[f"{k}_min"]  = float(np.nanmin(arrv))
        agg[f"{k}_max"]  = float(np.nanmax(arrv))
    return agg
