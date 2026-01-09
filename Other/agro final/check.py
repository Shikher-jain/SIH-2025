import numpy as np
import matplotlib.pyplot as plt

# Load structured NDVI .npy file
npy_path = '11.npy'
data = np.load(npy_path)

# Extract the 'NDVI' field from structured array
if data.dtype.names and 'NDVI' in data.dtype.names:
    ndvi = data['NDVI']
else:
    raise ValueError("NDVI field not found in structured array.")

# Show info
print(f"Extracted NDVI array: shape={ndvi.shape}, dtype={ndvi.dtype}")

# Plot NDVI heatmap
plt.figure(figsize=(6, 6))
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)  # NDVI ranges from -1 to 1
plt.colorbar(label='NDVI')
plt.title("NDVI Heatmap")
plt.axis('off')
plt.tight_layout()
plt.show()
