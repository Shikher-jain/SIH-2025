# modules/model_future.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, concatenate, Reshape

def build_future_predictor(img_shape=(64,64,6), future_days=60, seq_features=5, window_count=4, n_classes=3):
    """
    Model inputs:
      - current satellite image (H,W,C)
      - future weather sequence (future_days, seq_features)
    Outputs:
      - for each of window_count windows: n_classes softmax probabilities
    Implementation detail:
      - LSTM processes weather sequence -> vector
      - CNN processes image -> vector
      - Combine and predict window_count * n_classes outputs -> reshape (window_count, n_classes)
    """
    # Image branch
    img_in = Input(shape=img_shape, name="img_input")
    x = Conv2D(32, (3,3), activation="relu", padding="same")(img_in)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)

    # Weather branch
    seq_in = Input(shape=(future_days, seq_features), name="seq_input")
    y = LSTM(64)(seq_in)
    y = Dense(64, activation="relu")(y)

    # Combine
    z = concatenate([x,y])
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.3)(z)
    # predict window_count * n_classes numbers
    out_units = window_count * n_classes
    out = Dense(out_units, activation="linear")(z)  # linear first
    out = Reshape((window_count, n_classes))(out)
    out = tf.keras.activations.softmax(out, axis=-1)  # softmax across classes

    model = Model([img_in, seq_in], out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
