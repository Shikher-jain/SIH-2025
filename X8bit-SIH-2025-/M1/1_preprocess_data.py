# 1_preprocess_data.py

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load hyperspectral image and labels
image = loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
labels = loadmat('Indian_pines_gt.mat')['indian_pines_gt']

print("Image shape:", image.shape)
print("Labels shape:", labels.shape)

# Flatten image and labels
X = image.reshape(-1, image.shape[2])
y = labels.flatten()

# Remove unlabeled pixels
mask = y > 0
X = X[mask]
y = y[mask]

# Normalize spectral data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save numpy arrays
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Preprocessing done and files saved!")
