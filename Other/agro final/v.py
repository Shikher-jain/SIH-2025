import rasterio
import matplotlib.pyplot as plt

sensor_tif_path = r"C:\shikher_jain\SIH\test1\UttarPradesh_Test_Tiny_stacked_26bands_2025-09-27T22-54-02-251Z.tif"

with rasterio.open(sensor_tif_path) as src:
    # Read first three bands as RGB
    rgb = src.read([1, 2, 3])
    # Transpose to (height, width, channels)
    rgb_img = rgb.transpose(1, 2, 0)
    # Normalize for display
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

plt.figure(figsize=(8, 6))
plt.imshow(rgb_img)
plt.title("Sensor Data (RGB Composite)")
plt.axis("off")
plt.tight_layout()
plt.show()