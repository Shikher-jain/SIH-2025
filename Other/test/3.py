import rasterio
import numpy as np
import matplotlib.pyplot as plt

tif_path = r"C:\shikher_jain\SIH\test1\UttarPradesh_Test_Tiny_stacked_26bands_2025-09-27T22-54-02-251Z.tif"
with rasterio.open(tif_path) as src:
    data = src.read()
    profile = src.profile

# RGB Composite (bands 1,2,3)
rgb = data[:3].transpose(1,2,0)
rgb_img = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# NDVI
red = data[0]
nir = data[5]
ndvi = (nir - red) / (nir + red + 1e-6)

# Satellite-only health map
sat_health = np.clip(ndvi, 0, 1)

# Sensor layer (band 5)
sensor_layer = data[4]
# Complete health map
advanced_health = np.clip(ndvi*0.85 + sensor_layer*0.15, 0, 1)

# Stats
def stats(arr):
    return f"Mean: {arr.mean():.3f}\nMin: {arr.min():.3f}\nMax: {arr.max():.3f}"

plt.figure(figsize=(16, 12))
gs = plt.GridSpec(2, 3)

# RGB Composite
ax1 = plt.subplot(gs[0,0])
ax1.imshow(rgb_img)
ax1.set_title("RGB Composite")
ax1.axis('off')

# NDVI Heatmap
ax2 = plt.subplot(gs[0,1])
im2 = ax2.imshow(ndvi, cmap='RdYlGn')
ax2.set_title(f"NDVI (Mean: {ndvi.mean():.3f})")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.axis('off')

# Complete Health Map
ax3 = plt.subplot(gs[0,2])
im3 = ax3.imshow(advanced_health, cmap='RdYlGn')
ax3.set_title("Complete Health Map\n(Satellite + Sensor)")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
ax3.axis('off')

# Satellite Only Health Map
ax4 = plt.subplot(gs[1,0])
im4 = ax4.imshow(sat_health, cmap='RdYlGn')
ax4.set_title("Satellite Only Health Map\n(No Sensor Data)")
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
ax4.axis('off')

# Stats for Complete Health Map
ax5 = plt.subplot(gs[1,1])
ax5.text(0.1, 0.5, stats(advanced_health), fontsize=14)
ax5.set_title("Complete Map Statistics")
ax5.axis('off')

# Formula Panel
ax6 = plt.subplot(gs[1,2])
formula = "Complete Health = NDVI*0.6 + Sensor*0.4\nNDVI = (NIR - RED)/(NIR + RED)"
ax6.text(0.1, 0.5, formula, fontsize=14)
ax6.set_title("Complete Formula Panel")
ax6.axis('off')

plt.suptitle("Cell cell_1_1 - Health Status Report", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()