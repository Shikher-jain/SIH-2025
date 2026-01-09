import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.warp import reproject

# File paths
satellite_file = r"C:\shikher_jain\SIH\agro final\data\cell_1_1\test-1_satellite_2025-09-18_2025-09-23T13-55-47-115Z.tif"
sensor_file = r"C:\shikher_jain\SIH\agro final\data\cell_1_1\test-1_sensor_2025-09-23T13-56-05-971Z.tif"
merged_file = r"merged_analysis.tif"

# --- Function to normalize bands for visualization ---
def normalize(band):
    band = band.astype('float32')
    return (band - band.min()) / (band.max() - band.min() + 1e-6)

# --- 1. Satellite image ---
with rasterio.open(satellite_file) as src:
    satellite_rgb = np.stack([normalize(src.read(i)) for i in [1, 2, 3,4,5]], axis=-1)

# --- 2. Sensor image before resample ---
with rasterio.open(sensor_file) as src:
    sensor_rgb = np.stack([normalize(src.read(i)) for i in range(1,21)], axis=-1)

# --- 3. Sensor image after resample ---
with rasterio.open(sensor_file) as sensor_src, rasterio.open(satellite_file) as sat_src:
    target_width = sat_src.width
    target_height = sat_src.height
    target_transform = sat_src.transform
    target_crs = sat_src.crs

    sensor_resampled = []
    for i in [1, 2, 3]:
        src_data = sensor_src.read(i)
        dst_data = np.empty((target_height, target_width), dtype=src_data.dtype)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=sensor_src.transform,
            src_crs=sensor_src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        sensor_resampled.append(normalize(dst_data))
    sensor_resampled_rgb = np.stack(sensor_resampled, axis=-1)

# --- 4. Merged image preview ---
with rasterio.open(merged_file) as src:
    merged_rgb = np.stack([normalize(src.read(i)) for i in [1, 2, 3]], axis=-1)

# --- Plotting all 4 images ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(satellite_rgb)
axes[0, 0].set_title("🛰️ Satellite Image (before merge)")
axes[0, 0].axis("off")

axes[0, 1].imshow(sensor_rgb)
axes[0, 1].set_title("🧪 Sensor Image (before resample)")
axes[0, 1].axis("off")

axes[1, 0].imshow(sensor_resampled_rgb)
axes[1, 0].set_title("🧪 Sensor Image (after resample)")
axes[1, 0].axis("off")

axes[1, 1].imshow(merged_rgb)
axes[1, 1].set_title("🧬 Merged Image (final output)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
