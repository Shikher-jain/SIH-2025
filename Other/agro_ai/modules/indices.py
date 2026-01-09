# modules/indices.py
import numpy as np
from PIL import Image

def safe_div(a, b, eps=1e-10):
    return (a.astype(float)) / (b.astype(float) + eps)

def compute_indices_from_image(img_array):
    """
    img_array: H x W x B
    Band order expected (if 6-band): [Blue, Green, Red, NIR, SWIR1, SWIR2]
    Returns mean/std of common indices.
    """
    img = np.array(img_array, dtype=float)
    H, W, B = img.shape
    if B < 6:
        pad = np.zeros((H, W, 6 - B))
        img = np.concatenate([img, pad], axis=2)
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    nir = img[:,:,3]
    swir1 = img[:,:,4]
    swir2 = img[:,:,5]
    eps = 1e-10
    
    ndvi = safe_div(nir - red, nir + red, eps)
    savi = 1.5 * safe_div(nir - red, nir + red + 0.5, eps)
    gndvi = safe_div(nir - green, nir + green, eps)
    evi = 2.5 * safe_div(nir - red, nir + 6*red - 7.5*blue + 1, eps)
    si = safe_div(swir1 - nir, swir1 + nir, eps)
    ndsi = safe_div(swir1 - green, swir1 + green, eps)
    bsi = safe_div((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue), eps)

    def clean(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    out = {
        "NDVI_mean": float(np.mean(clean(ndvi))),
        "NDVI_std": float(np.std(clean(ndvi))),
        "SAVI_mean": float(np.mean(clean(savi))),
        "SAVI_std": float(np.std(clean(savi))),
        "GNDVI_mean": float(np.mean(clean(gndvi))),
        "GNDVI_std": float(np.std(clean(gndvi))),
        "EVI_mean": float(np.mean(clean(evi))),
        "EVI_std": float(np.std(clean(evi))),
        "SI_mean": float(np.mean(clean(si))),
        "SI_std": float(np.std(clean(si))),
        "NDSI_mean": float(np.mean(clean(ndsi))),
        "NDSI_std": float(np.std(clean(ndsi))),
        "BSI_mean": float(np.mean(clean(bsi))),
        "BSI_std": float(np.std(clean(bsi))),
        "bands_count": int(B)
    }
    return out

def resize_image(img_array, target=(128,128)):
    H,W,B = img_array.shape
    out = np.zeros((target[0], target[1], B), dtype=float)
    for i in range(B):
        band = img_array[:,:,i]
        band_min, band_max = np.nanmin(band), np.nanmax(band)
        if band_max - band_min < 1e-6:
            scaled = (band * 0).astype('uint8')
        else:
            scaled = ((band - band_min) / (band_max - band_min) * 255.0).astype('uint8')
        pil = Image.fromarray(scaled)
        pil = pil.resize(target, resample=Image.BILINEAR)
        arr = np.asarray(pil).astype(float) / 255.0
        out[:,:,i] = arr
    return out
