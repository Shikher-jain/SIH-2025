import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1️⃣ Load GeoTIFF & Preprocess
# ---------------------------

# Path to your GeoTIFF
tif_path = r"C:\shikher_jain\SIH\test1\UttarPradesh_Test_Tiny_stacked_26bands_2025-09-27T22-54-02-251Z.tif"

# Read first 6 bands using rasterio
with rasterio.open(tif_path) as src:
    data = src.read()[:6]  # shape: (bands, height, width)
    profile = src.profile

# Convert to HWC format (height, width, channels)
img_hwc = np.transpose(data, (1, 2, 0))  # shape: (H, W, 6)

# Normalize to [0, 1]
img_hwc_norm = (img_hwc - img_hwc.min()) / (img_hwc.max() - img_hwc.min())

# Resize image to 256x256 for model input
resized_img = tf.image.resize(img_hwc_norm, (256, 256)).numpy()
input_img = np.expand_dims(resized_img, axis=0)  # shape: (1, 256, 256, 6)

# ---------------------------
# 2️⃣ Define CNN Model
# ---------------------------

def cnn_model(input_shape=(256, 256, 6)):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = cnn_model(input_shape=(256, 256, 6))

# ---------------------------
# 3️⃣ Predict Wheat Presence
# ---------------------------

# ⚠️ Assumption: Model is not trained yet — result is random
wheat_prob = model.predict(input_img)[0][0]
print(f"Wheat Probability: {wheat_prob:.2f}")

# ---------------------------
# 4️⃣ Vegetation Indices & Health Map (Only if Wheat is Likely Present)
# ---------------------------

if wheat_prob > 0.25:
    print("✅ Wheat detected in the image!")

    # Extract bands from original image (512x512 resolution)
    B, G, R, NIR, SWIR1, SWIR2 = img_hwc.transpose(2, 0, 1)
    eps = 1e-6

    # Vegetation Indices
    NDVI = (NIR - R) / (NIR + R + eps)
    SAVI = ((NIR - R) / (NIR + R + 0.5 + eps)) * 1.5
    GNDVI = (NIR - G) / (NIR + G + eps)
    NDWI = (G - NIR) / (G + NIR + eps)
    MSI = SWIR1 / (NIR + eps)

    # Simple wheat mask (whole image assumed wheat, based on classification)
    wheat_mask = np.ones_like(NDVI, dtype=np.float32)

    # Apply mask
    NDVI *= wheat_mask
    SAVI *= wheat_mask
    GNDVI *= wheat_mask
    NDWI *= wheat_mask
    MSI *= wheat_mask

    NDVI_mean = np.nanmean(NDVI)

    # Weighted health map
    health_map = (
        0.25 * NDVI +
        0.15 * SAVI +
        0.10 * NDWI +
        0.10 * GNDVI +
        0.10 * MSI +
        0.30 * NDVI_mean
    )
    health_map = np.clip(health_map, 0, 1)

    # ---------------------------
    # 5️⃣ Show Health Map
    # ---------------------------
    plt.figure(figsize=(12, 6))
    plt.imshow(health_map, cmap='RdYlGn')
    plt.colorbar(label='Wheat Health Score')
    plt.title("🌾 Wheat Health Map (CNN-Based Detection)")
    plt.axis('off')
    plt.show()

else:
    print("❌ No wheat detected in the image.")
