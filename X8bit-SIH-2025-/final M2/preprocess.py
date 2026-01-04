import numpy as np
import cv2
import rasterio


def preprocess_image(img_path, target_size=(128, 128)):

    # 1. Load multispectral image (TIFF with multiple bands)     
    with rasterio.open(img_path) as src:
        # Read all bands into (C, H, W)
        img = src.read()
        # Reorder to (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    # 2. Normalize bands (0–1 scale)     
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

     
    # 3. Resize image for CNN
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 4. Compute NDVI
    # Formula: (NIR - Red) / (NIR + Red + 1e-6)
    # Assumption: Band order [Red, Green, Blue, NIR]
     
    red = img_resized[:, :, 0]
    nir = img_resized[:, :, 3]

    ndvi = (nir - red) / (nir + red + 1e-6)

    return img_resized, ndvi


    


    """

    Parameters
    ----------
    img_path : str
        Path to the input satellite/multispectral image (GeoTIFF).
    target_size : tuple
        Desired (height, width) for resizing image (default: 128x128).

    Returns
    -------
    img_resized : np.ndarray
        Preprocessed image with shape (H, W, C) where C = channels (RGB+NIR).
    ndvi : np.ndarray
        NDVI index values (2D array).
    """