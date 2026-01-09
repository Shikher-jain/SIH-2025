# file: validate_data.py
import os
import numpy as np
import rasterio

DATA_DIR = r".\data"

def validate_tif_files(data_dir=DATA_DIR):
    print("=== Validating .tif files ===")
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".tif"):
                fpath = os.path.join(root, f)
                try:
                    with rasterio.open(fpath) as src:
                        arr = src.read()
                        print(f"[OK] {fpath} | shape={arr.shape}, dtype={arr.dtype}")
                except Exception as e:
                    print(f"[BROKEN] {fpath} -> {e}")

def ensure_float32(*arrays):
    """Convert arrays to np.float32 and check for bad values."""
    fixed = []
    for i, arr in enumerate(arrays, start=1):
        if arr is None:
            raise ValueError(f"Array {i} is None!")
        arr = np.array(arr, dtype=np.float32)  # enforce numeric
        if np.isnan(arr).any():
            print(f"[WARN] NaN values found in array {i}, replacing with 0")
            arr = np.nan_to_num(arr)
        if arr.dtype != np.float32:
            print(f"[FIX] Converted array {i} to float32")
        fixed.append(arr)
    return fixed

if __name__ == "__main__":
    # Run .tif validation
    validate_tif_files()
