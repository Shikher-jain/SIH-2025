import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cnn_embedding(img_shape=(64,64,6)):
    # expects HWC
    inputs = layers.Input(shape=img_shape, name='img_in')
    x = layers.Conv2D(16,3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32,3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    model = Model(inputs, x, name='cnn_embed')
    return model

