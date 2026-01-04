import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from tensorflow.keras import layers, models

# STEP 1: Load Hyperspectral Data (simulate with random array)
hyperspectral_img = np.random.rand(100, 100, 50)   # Dummy image

# Flatten for PCA/ICA
pixels = hyperspectral_img.reshape(-1, 50)

# PCA / ICA
pca = PCA(n_components=5).fit_transform(pixels)
ica = FastICA(n_components=5).fit_transform(pixels)

# STEP 2: Vegetation Indices
nir = hyperspectral_img[:, :, 40]
red = hyperspectral_img[:, :, 20]
blue = hyperspectral_img[:, :, 10]

ndvi = (nir - red) / (nir + red + 1e-6)
savi = ((nir - red) / (nir + red + 0.5)) * 1.5
pri = (blue - red) / (blue + red + 1e-6)

# Save vegetation indices for visualization.py
np.save("ndvi.npy", ndvi)
np.save("savi.npy", savi)
np.save("pri.npy", pri)

# STEP 3: Sensor Data
sensor_data = pd.DataFrame({
    "soil_moisture": np.random.rand(100),
    "temp": np.random.rand(100)*40,
    "humidity": np.random.rand(100)*100,
    "leaf_wetness": np.random.rand(100)
})

sensor_features = sensor_data.mean().values

# STEP 4: CNN Model
veg_indices = np.stack([ndvi, savi, pri], axis=-1)   # (100, 100, 3)
X = np.expand_dims(veg_indices, axis=0)  # (1,100,100,3)

cnn_model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(100, 100, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])

cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

y_fake = np.array([1])  # dummy label
cnn_model.fit(X, y_fake, epochs=1, verbose=0)

prediction = cnn_model.predict(X, verbose=0)[0][0]

result = {
    "stress_probability": float(prediction),
    "avg_soil_moisture": float(sensor_features[0]),
    "avg_temp": float(sensor_features[1]),
    "ndvi_mean": float(ndvi.mean()),
    "savi_mean": float(savi.mean()),
    "pri_mean": float(pri.mean())
}

print("final Output:", result)
