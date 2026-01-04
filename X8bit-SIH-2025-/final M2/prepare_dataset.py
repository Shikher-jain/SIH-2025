from preprocess import preprocess_image
import numpy as np
import os

X, y = [], []
 
# Healthy crops = 0, Unhealthy crops = 1
for label, folder in enumerate(["data/train/healthy", "data/train/unhealthy"]):
    for file in os.listdir(folder):
        if file.endswith(".tif"):
            img, ndvi = preprocess_image(os.path.join(folder, file))
            X.append(img[:, :, :3])  # CNN ke liye sirf RGB
            y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("Preprocessing complete. X_train.npy & y_train.npy saved.")