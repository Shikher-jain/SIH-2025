import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load multi-band .tif file
with rasterio.open('sentinel_multiband.tif') as src:
    image = src.read()  # shape: (bands, height, width)
    profile = src.profile

print("Shape:", image.shape)  # e.g., (10, 512, 512)


nir = image[7].astype('float32')   # Band 8
red = image[3].astype('float32')   # Band 4

ndvi = (nir - red) / (nir + red + 1e-6)  # avoid divide by zero

# Plot NDVI
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI Map")
plt.show()
