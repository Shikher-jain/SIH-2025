import io
import numpy as np
from PIL import Image


def compute_ndvi_from_bands(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED) safely."""
    nir = np.array(nir, dtype=np.float32)
    red = np.array(red, dtype=np.float32)
    denom = (nir + red)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / denom
    ndvi[np.isinf(ndvi)] = np.nan
    return ndvi


def normalize_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """Normalize NDVI to 0..1, preserving NaNs."""
    nd = np.array(ndvi, dtype=np.float32)
    mask = np.isfinite(nd)
    if not np.any(mask):
        return np.full_like(nd, np.nan, dtype=np.float32)
    mn = np.nanmin(nd)
    mx = np.nanmax(nd)
    if mx - mn == 0:
        # constant array
        out = np.zeros_like(nd, dtype=np.float32)
        out[~mask] = np.nan
        return out
    out = (nd - mn) / (mx - mn)
    out[~mask] = np.nan
    return out


def adjust_thresholds_by_yield(low: float, high: float, yield_value: float = None, factor: float = 0.05):
    """Adjust thresholds slightly based on yield_value. If yield_value is None, return unchanged."""
    if yield_value is None:
        return low, high
    try:
        y = float(yield_value)
    except Exception:
        return low, high
    # simple rule: higher yield -> stricter (increase thresholds), lower yield -> decrease
    shift = (y - 1.0) * factor  # assume yield around 1.0 is baseline; user can pick scale
    return max(0.0, low + shift), min(1.0, high + shift)


def ndvi_to_rgba_image(ndvi_norm: np.ndarray, low: float = 0.33, high: float = 0.66) -> Image.Image:
    """Convert normalized NDVI (0..1, NaN for missing) to an RGBA PIL Image.

    - values < low -> red
    - low <= values < high -> yellow
    - values >= high -> green
    - NaN -> transparent
    """
    arr = np.array(ndvi_norm, dtype=np.float32)
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Transparent by default (alpha=0)
    rgba[..., 3] = 0

    mask_valid = np.isfinite(arr)
    if np.any(mask_valid):
        valid = arr[mask_valid]
        # classify
        red_mask = (valid < low)
        yellow_mask = (valid >= low) & (valid < high)
        green_mask = (valid >= high)

        # assign colors
        rgba[..., 3][mask_valid] = 255
        # We need to place colors back into full arrays
        rgba_r = rgba[..., 0]
        rgba_g = rgba[..., 1]
        rgba_b = rgba[..., 2]

        rgba_r[mask_valid] = 0
        rgba_g[mask_valid] = 0
        rgba_b[mask_valid] = 0

        # create full boolean masks by filling into same shape
        full = np.zeros_like(arr, dtype=bool)
        full[mask_valid] = red_mask
        rgba[..., 0][full] = 255  # red

        full2 = np.zeros_like(arr, dtype=bool)
        full2[mask_valid] = yellow_mask
        rgba[..., 0][full2] = 255
        rgba[..., 1][full2] = 255

        full3 = np.zeros_like(arr, dtype=bool)
        full3[mask_valid] = green_mask
        rgba[..., 1][full3] = 255

    return Image.fromarray(rgba, mode='RGBA')


def rgba_image_to_png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    return bio.getvalue()


def ndvi_array_to_png(ndvi_array: np.ndarray, thresholds=(0.33, 0.66), yield_value: float = None) -> bytes:
    low, high = thresholds
    low, high = adjust_thresholds_by_yield(low, high, yield_value)
    norm = normalize_ndvi(ndvi_array)
    img = ndvi_to_rgba_image(norm, low=low, high=high)
    return rgba_image_to_png_bytes(img)
