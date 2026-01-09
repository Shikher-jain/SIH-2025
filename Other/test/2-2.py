import rasterio
from rasterio.enums import Resampling
import numpy as np
from affine import Affine

# 1. Define File Paths and Target Parameters
HR_FILE = r"C:\shikher_jain\SIH\agro final\data\cell_1_1\test-1_satellite_2025-09-18_2025-09-23T13-55-47-115Z.tif"  # 100m
LR_FILE = r"C:\shikher_jain\SIH\agro final\data\cell_1_1\test-1_sensor_2025-09-23T13-56-05-971Z.tif"      # 1000m
OUTPUT_FILE = "merged_128x128.tif"
TARGET_SHAPE = (128, 128)  # (Height, Width)

print(f"Goal: Resizing and merging both files to {TARGET_SHAPE[0]}x{TARGET_SHAPE[1]} pixels.")

# --- 2. Process High-Resolution (100m Satellite) Data ---
# This is DOWN SCALED (from its original size to 128x128)
with rasterio.open(HR_FILE) as hr_src:
    HR_PROFILE = hr_src.profile
    hr_count = hr_src.count
    
    # Use CUBIC SPLINE for downscaling: minimizes data loss of spatial details.
    HR_DATA = hr_src.read(
        out_shape=(hr_count, *TARGET_SHAPE),
        resampling=Resampling.cubic_spline 
    )
    print(f"Satellite data (100m) resized using Cubic Spline.")

# --- 3. Process Low-Resolution (1000m Sensor) Data ---
# This is UP SCALED (to fit the 128x128 grid)
with rasterio.open(LR_FILE) as lr_src:
    lr_count = lr_src.count

    # Use BILINEAR for upscaling: provides smooth value estimation.
    LR_DATA = lr_src.read(
        out_shape=(lr_count, *TARGET_SHAPE),
        resampling=Resampling.bilinear 
    )
    print(f"Sensor data (1000m) resized using Bilinear.")

# --- 4. Stack Data Arrays ---
# Combine the bands from both sources into one array.
merged_data_final = np.concatenate([HR_DATA, LR_DATA], axis=0)
total_bands = merged_data_final.shape[0]
print(f"\nSuccessfully merged. Final array shape: (Bands: {total_bands}, Height: 128, Width: 128)")

# --- 5. Write the Merged GeoTIFF (with Correct Georeferencing) ---

# Update the metadata based on the high-resolution file's original extent.
out_profile = HR_PROFILE.copy()

# Calculate the new pixel size to ensure the 128x128 grid spans the original geographic area.
original_pixel_width = HR_PROFILE['transform'].a
original_pixel_height = abs(HR_PROFILE['transform'].e)

# New pixel size = (Original Width in Pixels * Original Pixel Size) / New Width in Pixels
new_pixel_width = (HR_PROFILE['width'] * original_pixel_width) / TARGET_SHAPE[1]
new_pixel_height = (HR_PROFILE['height'] * original_pixel_height) / TARGET_SHAPE[0]

# Create the new affine transform (essential for correct georeferencing)
new_transform = Affine(new_pixel_width, 0.0, HR_PROFILE['transform'].c,
                       0.0, -new_pixel_height, HR_PROFILE['transform'].f)

out_profile.update({
    'count': total_bands,
    'height': merged_data_final.shape[1],
    'width': merged_data_final.shape[2],
    'transform': new_transform,
    'dtype': merged_data_final.dtype,
    'compress': 'lzw'
})

with rasterio.open(OUTPUT_FILE, 'w', **out_profile) as dst:
    dst.write(merged_data_final)
    
print(f"\nFinal merged GeoTIFF saved as: **{OUTPUT_FILE}**")