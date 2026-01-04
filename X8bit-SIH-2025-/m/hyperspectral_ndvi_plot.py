# hyperspectral_ndvi_plot.py

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load hyperspectral image
data = loadmat('Indian_pines_corrected.mat')
image = data['indian_pines_corrected']   # shape: (145, 145, 220)

print("Image shape:", image.shape)

# Step 2: Extract bands for NDVI
# Assume: NIR = band 80, Red = band 50 (just an example)
nir = image[:, :, 80].astype(np.float32)
red = image[:, :, 50].astype(np.float32)

ndvi = (nir - red) / (nir + red + 1e-6)

# Step 3: Plot NDVI
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI Map (Hyperspectral)")
plt.show()

# Step 4: Plot spectral signature of a pixel
row, col = 60, 70
spectrum = image[row, col, :]
plt.figure()
plt.plot(spectrum)
plt.title(f"Spectral Signature at ({row},{col})")
plt.xlabel("Band Number")
plt.ylabel("Reflectance")
plt.show()
