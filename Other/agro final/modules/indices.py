import numpy as np
import rasterio

def compute_indices_from_satellite(sat_file):
    """
    Compute NDVI, EVI, GNDVI, NDWI, etc. from satellite bands.
    """
    with rasterio.open(sat_file) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] < 4:
        raise ValueError("Expected >=4 bands")
    B, G, R, NIR = arr[:4]
    eps = 1e-6

    indices = {}
    indices['NDVI'] = (NIR - R)/(NIR + R + eps)
    indices['EVI']  = 2.5*(NIR-R)/(NIR + 6*R - 7.5*B + 1 + eps)
    indices['GNDVI'] = (NIR - G)/(NIR + G + eps)
    indices['NDWI'] = (G - NIR)/(G + NIR + eps)

    agg = {}
    for k,v in indices.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"]  = float(np.std(v))
        agg[f"{k}_min"]  = float(np.min(v))
        agg[f"{k}_max"]  = float(np.max(v))
    return agg

def compute_indices_from_sensor(sensor_file):
    """
    N, P, ECe, OC, pH -> aggregated features
    """
    with rasterio.open(sensor_file) as src:
        arr = src.read().astype(np.float32)
    # assuming bands order: N,P,ECe,OC,pH
    bands = ['N','P','ECe','OC','pH']
    agg = {}
    for i,name in enumerate(bands):
        b = arr[i]
        agg[f"{name}_mean"] = float(np.mean(b))
        agg[f"{name}_std"] = float(np.std(b))
        agg[f"{name}_min"] = float(np.min(b))
        agg[f"{name}_max"] = float(np.max(b))
    return agg
