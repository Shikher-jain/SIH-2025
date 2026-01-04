import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def build_cnn(input_shape=(128,128,3), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# Example training loop
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

model = build_cnn()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save("crop_health_model.h5")
