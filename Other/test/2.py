import matplotlib.pyplot as plt
import rasterio
import numpy as np

# Path to merged GeoTIFF
tif_path = r'C:\shikher_jain\SIH\test1\UttarPradesh_Test_Tiny_stacked_26bands_2025-09-27T22-54-02-251Z.tif'

with rasterio.open(tif_path) as src:
    # Read all bands at once
    data = src.read()  # Shape: (26, height, width)
    
print(f"Data shape: {data.shape}")  # (bands, H, W)

red_band_index = 3   # Example: Band 4 is Red (0-indexed)
nir_band_index = 7   # Example: Band 8 is NIR

red = data[red_band_index]
nir = data[nir_band_index]

ndvi = (nir - red) / (nir + red + 1e-6)

# Example: Use all 26 bands as channels for CNN
# Shape needed: (height, width, channels)
input_stack = np.transpose(data, (1, 2, 0))
print(f"Stack shape: {input_stack.shape}")  # (H, W, 26)

# Example: weighted NDVI + first few sensor bands
predicted_health = ndvi*0.6 + 0.4*input_stack[:,:,10]  # Band 11 = soil moisture
predicted_health = np.clip(predicted_health, 0, 1)


plt.figure(figsize=(10,8))
plt.imshow(predicted_health, cmap='RdYlGn')
plt.colorbar(label='Crop Health / Yield')
plt.title('Advanced Wheat Crop Heatmap (Merged 26 Bands)')
plt.axis('off')
plt.show()
