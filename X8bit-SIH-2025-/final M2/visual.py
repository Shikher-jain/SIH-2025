import numpy as np
import plotly.express as px
from preprocess import preprocess_image

def visualize_indices(img_path):
    # Preprocess image
    img, ndvi = preprocess_image(img_path)

    # Extract bands
    red = img[:, :, 0]
    green = img[:, :, 1]
    nir = img[:, :, 3]

    # Compute SAVI
    savi = (1.5 * (nir - red)) / (nir + red + 0.5)

    # Compute PRI
    pri = (green - red) / (green + red + 1e-6)

    # Plot NDVI
    fig_ndvi = px.imshow(ndvi, color_continuous_scale="RdYlGn", origin="upper")
    fig_ndvi.update_layout(title="NDVI (Normalized Difference Vegetation Index)")
    fig_ndvi.show()

    # Plot SAVI
    fig_savi = px.imshow(savi, color_continuous_scale="YlGn", origin="upper")
    fig_savi.update_layout(title="SAVI (Soil Adjusted Vegetation Index)")
    fig_savi.show()

    # Plot PRI
    fig_pri = px.imshow(pri, color_continuous_scale="Viridis", origin="upper")
    fig_pri.update_layout(title="PRI (Photochemical Reflectance Index)")
    fig_pri.show()


if __name__ == "__main__":
    # Example image (multispectral GeoTIFF with Red, Green, Blue, NIR bands)
    sample_path = "sample_image.tif"
    visualize_indices(sample_path)
