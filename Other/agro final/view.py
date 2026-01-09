import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
# Display all .tif files in the folder in a single window
folder = os.path.dirname(__file__)
folder = 'vv'
tif_files = [f for f in os.listdir(folder) if f.lower().endswith('.tif')]

if not tif_files:
    raise FileNotFoundError("No .tif files found in the folder.")

num_files = len(tif_files)
cols = min(3, num_files)
rows = (num_files + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1 or cols == 1:
    axes = np.atleast_2d(axes)

max_dim = 1000

for idx, tif_file in enumerate(tif_files):
    ax = axes.flat[idx]
    try:
        with rasterio.open(os.path.join(folder, tif_file)) as src:
            band = src.read(1)
        if band.shape[0] > max_dim or band.shape[1] > max_dim:
            factor = max(band.shape[0] // max_dim, band.shape[1] // max_dim)
            band_small = band[::factor, ::factor]
        else:
            band_small = band
        im = ax.imshow(band_small, cmap="RdYlGn")
        ax.set_title(tif_file)
        ax.axis("off")
    except Exception as e:
        ax.set_title(f"Error: {tif_file}")
        ax.axis("off")
        print(f"Error reading {tif_file}: {e}")

# Hide any unused subplots
for i in range(num_files, rows*cols):
    axes.flat[i].axis('off')

fig.tight_layout()
plt.show()