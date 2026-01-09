from tensorflow.keras import layers, Model, Input
import tensorflow as tf

def build_cnn_lstm_model(img_shape, seq_len=24, seq_features=5, n_classes=3):
    # CNN for spatial
    img_input = Input(shape=img_shape)
    x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(img_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64,activation='relu')(x)

    # LSTM for weather
    seq_input = Input(shape=(seq_len, seq_features))
    y = layers.LSTM(32)(seq_input)
    y = layers.Dense(32,activation='relu')(y)

    # Merge
    z = layers.Concatenate()([x,y])
    z = layers.Dense(64,activation='relu')(z)
    out = layers.Dense(n_classes,activation='softmax')(z)

    model = Model(inputs=[img_input, seq_input], outputs=out)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
