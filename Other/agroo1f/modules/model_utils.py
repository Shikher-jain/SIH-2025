import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, concatenate

def build_cnn_lstm_model(img_shape=(64,64,6), seq_len=24, n_classes=3):
    # CNN for images
    img_input = Input(shape=img_shape)
    x = Conv2D(32,(3,3),activation="relu",padding="same")(img_input)
    x = MaxPooling2D()(x)
    x = Conv2D(64,(3,3),activation="relu",padding="same")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)

    # LSTM for weather sequence
    seq_input = Input(shape=(seq_len,5))
    y = LSTM(32)(seq_input)

    # Combine
    z = concatenate([x,y])
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)
    out = Dense(n_classes, activation="softmax")(z)

    model = Model([img_input, seq_input], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
