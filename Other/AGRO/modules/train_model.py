import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_cnn_lstm(input_img_shape=(64, 64, 6), seq_shape=(24,5), tab_shape=5, num_classes=3):
    # --- Image branch (CNN) ---
    img_input = layers.Input(shape=input_img_shape, name="image_input")
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    # --- Sequence branch (LSTM) ---
    seq_input = layers.Input(shape=seq_shape, name="seq_input")
    y = layers.LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(1e-4))(seq_input)
    y = layers.Dropout(0.5)(y)

    # --- Tabular branch ---
    tab_input = layers.Input(shape=(tab_shape,), name="tab_input")
    z = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(tab_input)
    z = layers.Dropout(0.25)(z)

    # --- Combine ---
    combined = layers.concatenate([x, y, z])
    combined = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(num_classes, activation='softmax')(combined)

    model = models.Model(inputs=[img_input, seq_input, tab_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_lstm(X_img_tr, X_seq_tr, X_tab_tr, y_tr,
                   X_img_te, X_seq_te, X_tab_te, y_te,
                   epochs=50, batch_size=16, save_path="outputs/models/cnn_lstm.keras"):

    model = build_cnn_lstm(input_img_shape=X_img_tr.shape[1:],
                            seq_shape=X_seq_tr.shape[1:],
                            tab_shape=X_tab_tr.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)

    history = model.fit(
        [X_img_tr, X_seq_tr, X_tab_tr], y_tr,
        validation_data=([X_img_te, X_seq_te, X_tab_te], y_te),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    print(f"✅ Model saved at {save_path}")
    return model, history
