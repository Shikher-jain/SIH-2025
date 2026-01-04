# multispectral_ndvi_plot.py

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load multi-band .tif image
with rasterio.open('sentinel_multiband.tif') as src:
    image = src.read()  # shape: (bands, height, width)

print("Image shape (bands, height, width):", image.shape)

# Step 2: Extract Red and NIR bands
# Sentinel-2: Band 4 = Red (index 3), Band 8 = NIR (index 7)
red = image[3].astype(np.float32)
nir = image[7].astype(np.float32)

# Step 3: Calculate NDVI
ndvi = (nir - red) / (nir + red + 1e-6)

# Step 4: Plot NDVI
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI Map (Multispectral)")
plt.show()
