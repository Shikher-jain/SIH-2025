import rasterio
import numpy as np
import json

def compute_indices(sat_file, out_json, out_npy):
    with rasterio.open(sat_file) as src:
        arr = src.read()  # shape = (bands, H, W)
        profile = src.profile
        band_count = arr.shape[0]

    # --- Assign bands dynamically (depending on available count) ---
    red = arr[2] if band_count >= 3 else None
    green = arr[1] if band_count >= 2 else None
    blue = arr[0] if band_count >= 1 else None
    nir = arr[3] if band_count >= 4 else None   # sometimes 4th band = NIR
    swir = arr[4] if band_count >= 5 else None  # if available

    indices = {}

    # NDVI
    if nir is not None and red is not None:
        ndvi = (nir - red) / (nir + red + 1e-6)
        indices["NDVI"] = ndvi.tolist()
    else:
        indices["NDVI"] = None

    # GNDVI (NIR - Green) / (NIR + Green)
    if nir is not None and green is not None:
        gndvi = (nir - green) / (nir + green + 1e-6)
        indices["GNDVI"] = gndvi.tolist()
    else:
        indices["GNDVI"] = None

    # NDWI (Green - NIR) / (Green + NIR)
    if green is not None and nir is not None:
        ndwi = (green - nir) / (green + nir + 1e-6)
        indices["NDWI"] = ndwi.tolist()
    else:
        indices["NDWI"] = None

    # Save as JSON
    with open(out_json, "w") as f:
        json.dump(indices, f)

    # Save as NumPy
    np.save(out_npy, indices)

    print(f"Saved indices → {out_json}, {out_npy}")


if __name__ == "__main__":

    # sat_file = r"C:\shikher_jain\SIH\SIH 2025\agro_ai\data\cell_3_50\satellite.tif"
    sat_file = "C:/shikher_jain/SIH/SIH 2025/agro_ai/data/cell_1_1/test-1_satellite_2025-09-18_2025-09-23T13-55-47-115Z.tif"
    out_json = r"C:\shikher_jain\SIH\SIH 2025\agro_ai\outputs\cell_3_50_indices.json"
    out_npy  = r"C:\shikher_jain\SIH\SIH 2025\agro_ai\outputs\cell_3_50_indices.npy"

    compute_indices(sat_file, out_json, out_npy)
