import rasterio
import numpy as np
import matplotlib.pyplot as plt

def plot_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        red = src.read(3)
        green = src.read(2)
        blue = src.read(1)
        rgb = np.stack([red, green, blue], axis=-1).astype('float32')
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb)
    plt.title("Sentinel-2 True Color (RGB)")
    plt.axis("off")
    plt.show()

def plot_ndvi(tif_path):
    with rasterio.open(tif_path) as src:
        red = src.read(3).astype('float32')   # Band 4
        nir = src.read(5).astype('float32')   # Band 8
        ndvi = (nir - red) / (nir + red + 1e-9)
        ndvi = np.clip(ndvi, -1, 1)
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar(heatmap, label="NDVI")
    plt.title("NDVI Heatmap (Sentinel-2)")
    plt.axis("off")
    plt.show()

def plot_all_bands(tif_path):
    with rasterio.open(tif_path) as src:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i in range(6):
            band = src.read(i + 1).astype('float32')
            band = (band - band.min()) / (band.max() - band.min() + 1e-9)
            axes[i].imshow(band, cmap='gray')
            axes[i].set_title(f"Band {i + 1}")
            axes[i].axis("off")
        plt.suptitle("All 6 Sentinel-2 Bands")
        plt.tight_layout()
        plt.show()

def plot_false_color(tif_path):
    with rasterio.open(tif_path) as src:
        
        nir   = src.read(5)
        red   = src.read(3)
        green = src.read(2)

        rgb = np.stack([nir, red, green], axis=-1).astype('float32')
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)

    plt.figure(figsize=(8, 6))
    # plt.imshow(rgb ,cmap="RdYlGn")
    # heatmap = plt.imshow(rgb, cmap='RdYlGn')
    heatmap = plt.imshow(rgb, cmap='hot')
    plt.colorbar(heatmap, label="NDVI")
    plt.title("False Color Composite (NIR, Red, Green)")
    plt.axis("off")
    plt.show()

# ==== FILE PATH ====
tif_file_path = r"C:\shikher_jain\SIH\agro final\data\cell_1_1\test-1_satellite_2025-09-18_2025-09-23T13-55-47-115Z.tif"
# tif_file_path = r"C:\shikher_jain\SIH\agro final\Mb-1_sensor_area1_cell_01_2025-09-28T22-04-03-173Z.tif"

# ==== RUN ====
plot_rgb(tif_file_path)
plot_ndvi(tif_file_path)
plot_all_bands(tif_file_path)
plot_false_color(tif_file_path)
