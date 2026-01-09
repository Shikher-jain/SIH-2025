# 2_extract_features.py

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat('Indian_pines_corrected.mat')
image = data['indian_pines_corrected']

# Bands (example): NIR=80, Red=50
nir = image[:, :, 80].astype(float)
red = image[:, :, 50].astype(float)

ndvi = (nir - red) / (nir + red + 1e-6)

plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI Map")
plt.show()

