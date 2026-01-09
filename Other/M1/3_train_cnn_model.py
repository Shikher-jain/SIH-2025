# 3_train_cnn_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Reshape for CNN input: (samples, bands, 1, 1)
X_train = X_train.reshape(-1, 220, 1, 1)
X_test = X_test.reshape(-1, 220, 1, 1)

# Build CNN model
model = Sequential([
    Conv2D(8, (3,1), activation='relu', padding='same', input_shape=(220,1,1)),
    MaxPooling2D(pool_size=(2,1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(set(y_train)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Save model
model.save('cnn_model.h5')

print("Training complete and model saved!")
