import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ---------------------------
# 1️⃣ Function to compute all indices
# ---------------------------
def compute_indices_from_image(img_array):
    """
    Compute vegetation indices from multi-band image
    Input: HWC format
    Returns: dict of indices arrays
    """
    if img_array.shape[2] <= 6:
        img_array = np.transpose(img_array, (2,0,1))  # HWC -> CHW
    B, G, R, NIR, SWIR1, SWIR2 = img_array[:6]

    eps = 1e-6
    indices = {}
    indices['NDVI']    = (NIR - R) / (NIR + R + eps)
    indices['EVI']     = 2.5*(NIR - R)/(NIR + 6*R - 7.5*B + 1 + eps)
    indices['SAVI']    = ((NIR - R)/(NIR + R + 0.5 + eps))*1.5
    indices['GNDVI']   = (NIR - G)/(NIR + G + eps)
    indices['NDWI']    = (G - NIR)/(G + NIR + eps)
    indices['MSI']     = SWIR1/(NIR + eps)
    return indices

# ---------------------------
# 2️⃣ Load GeoTIFF (26 bands)
# ---------------------------
# tif_path = r"C:\shikher_jain\SIH\test1\UttarPradesh_Test_Tiny_stacked_26bands_2025-09-27T22-54-02-251Z.tif"

tif_path = r"C:\shikher_jain\SIH\test1\Test14-1_stacked_26bands_2025-09-28T11-23-46-570Z.tif"
# tif_path = r"C:\shikher_jain\SIH\test1\Test14-2_stacked_26bands_2025-09-28T11-29-23-800Z.tif"

with rasterio.open(tif_path) as src:
    data = src.read()           # CHW format: (26, H, W)
    profile = src.profile

# Convert to HWC for processing
img_hwc = np.transpose(data[:6], (1,2,0))  # Using first 6 bands

# ---------------------------
# 3️⃣ Compute all indices
# ---------------------------
indices = compute_indices_from_image(img_hwc)

# ---------------------------
# 4️⃣ Aggregate statistics for global context
# ---------------------------
indices_stats = {name: float(np.nanmean(arr)) for name, arr in indices.items()}

# ---------------------------
# 5️⃣ Compute wheat health map (weighted)
# ---------------------------
health_map = (
    0.25 * np.clip(indices['NDVI'], 0, 1) +
    0.15 * np.clip(indices['SAVI'], 0, 1) +
    0.10 * np.clip(indices['NDWI'], 0, 1) +
    0.10 * np.clip(indices['GNDVI'], 0, 1) +
    0.10 * np.clip(indices['MSI'], 0, 1) +
    0.30 * indices_stats['NDVI']  # Global mean NDVI as context
)
health_map = np.clip(health_map, 0, 1)

# # ---------------------------
# # 6️⃣ Save Raster Health Map (GeoTIFF)
# # ---------------------------
# output_path = r"C:\shikher_jain\SIH\outputs\wheat_health_map.tif"
# profile.update(dtype=rasterio.float32, count=1)

# with rasterio.open(output_path, 'w', **profile) as dst:
#     dst.write(health_map.astype(rasterio.float32), 1)

# print(f"Raster Wheat Health Map saved at: {output_path}")

# ---------------------------
# 7️⃣ Optional Visualization
# ---------------------------
plt.figure(figsize=(12,6))
plt.imshow(health_map, cmap='RdYlGn')
plt.colorbar(label='Wheat Health Score')
plt.title('Advanced Wheat Health Map (Multi-Index)')
plt.axis('off')
plt.show()

# ---------------------------
# 8️⃣ DDE / Multi-Month Integration (Example)
# ---------------------------
# Suppose multiple GeoTIFFs for Jan, Feb, Mar
tif_files = sorted(glob.glob(r"C:\shikher_jain\SIH\test1\*_stacked_26bands_*.tif"))
time_series_health = []

for f in tif_files:
    with rasterio.open(f) as src:
        data_ts = src.read()
    img_hwc = np.transpose(data_ts[:6], (1,2,0))
    indices_ts = compute_indices_from_image(img_hwc)
    indices_stats_ts = {name: float(np.nanmean(arr)) for name, arr in indices_ts.items()}
    health_map_ts = (
        0.25 * np.clip(indices_ts['NDVI'],0,1) +
        0.15 * np.clip(indices_ts['SAVI'],0,1) +
        0.10 * np.clip(indices_ts['NDWI'],0,1) +
        0.10 * np.clip(indices_ts['GNDVI'],0,1) +
        0.10 * np.clip(indices_ts['MSI'],0,1) +
        0.30 * indices_stats_ts['NDVI']
    )
    health_map_ts = np.clip(health_map_ts,0,1)
    time_series_health.append(health_map_ts)

# Plot multi-month NDVI / health evolution
months = ['Jan', 'Feb', 'Mar']
for month, hmap in zip(months, time_series_health):
    plt.figure(figsize=(8,6))
    plt.imshow(hmap, cmap='RdYlGn')
    plt.colorbar(label='Health Score')
    plt.title(f'Wheat Health Map - {month}')
    plt.axis('off')
    plt.show()
