import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, TimeDistributed, Conv2D, MaxPooling2D,
                                     Flatten, LSTM, Dense, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Load NDVI, sensor, and yield
# -----------------------------
def load_ndvi_sensor_yield(ndvi_folder, sensor_folder, yield_csv_path):
    """
    Assumes:
    - ndvi_folder has .npy files; each file may contain a batch of samples or one sample
    - sensor_folder similar
    - yield_csv has a row per sample, in the same order (or with an ID you can align)
    """
    # List and sort files so they align (important)
    ndvi_files = sorted([os.path.join(ndvi_folder, f) for f in os.listdir(ndvi_folder) if f.endswith('.npy')])
    sensor_files = sorted([os.path.join(sensor_folder, f) for f in os.listdir(sensor_folder) if f.endswith('.npy')])
    
    # Load all ndvi arrays and concatenate along first axis
    ndvi_list = [np.load(f) for f in ndvi_files]
    ndvi_data = np.concatenate(ndvi_list, axis=0)
    
    sensor_list = [np.load(f) for f in sensor_files]
    sensor_data = np.concatenate(sensor_list, axis=0)
    
    # Load yield CSV
    df = pd.read_csv(yield_csv_path)
    # Suppose yield column is named "yield" (you may need to change)
    yield_array = df['yield'].values
    
    # Optionally, align by ID if there is an “id” column; here we assume ordering matches
    return ndvi_data, sensor_data, yield_array

# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess_inputs(ndvi_data, sensor_data):
    """
    Normalize NDVI and standardize sensor data.
    ndvi_data shape: (num_samples, seq_len, H, W) or (num_samples, H, W) if no sequence
    sensor_data shape: (num_samples, seq_len, num_features) or (num_samples, num_features)
    """
    # Normalize NDVI to [0,1]
    ndvi_data = ndvi_data.astype('float32') / 255.0
    
    # If NDVI is 3D (no time), add a dummy time dimension
    if ndvi_data.ndim == 4:  # (N, H, W, maybe channel)
        # reshape to (N, 1, H, W)
        ndvi_data = np.expand_dims(ndvi_data, axis=1)
    # If there's no channel dim, add it
    if ndvi_data.ndim == 5:  # (N, seq_len, H, W, channels)
        pass
    elif ndvi_data.ndim == 4:  # (N, seq_len, H, W)
        ndvi_data = ndvi_data[..., np.newaxis]  # add channel dim
    
    # Standardize sensor
    # Flatten the time dimension and features for scaling, then reshape back
    ns = sensor_data.shape
    if sensor_data.ndim == 3:  # (N, seq_len, num_features)
        N, T, F = ns
        sensor_flat = sensor_data.reshape((N * T, F))
        scaler = StandardScaler()
        sensor_scaled_flat = scaler.fit_transform(sensor_flat)
        sensor_scaled = sensor_scaled_flat.reshape((N, T, F))
    elif sensor_data.ndim == 2:  # (N, F)
        scaler = StandardScaler()
        sensor_scaled = scaler.fit_transform(sensor_data)
    else:
        raise ValueError("sensor_data ndim not supported")
    
    return ndvi_data, sensor_scaled

# -----------------------------
# 3. Build CNN-LSTM Model
# -----------------------------
def build_cnn_lstm_model(ndvi_shape, sensor_shape):
    """
    ndvi_shape: (seq_len, H, W, channels)
    sensor_shape: (seq_len, num_features) or (num_features,) if no time
    """
    # NDVI pathway
    ndvi_input = Input(shape=ndvi_shape, name='ndvi_input')
    # Use TimeDistributed conv layers
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(ndvi_input)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    # LSTM to capture temporal features
    x = LSTM(64, name='ndvi_lstm')(x)
    
    # Sensor pathway
    sensor_input = Input(shape=sensor_shape, name='sensor_input')
    if len(sensor_shape) == 2:
        # sequence sensor
        y = LSTM(64, name='sensor_lstm')(sensor_input)
    else:
        # no time axis: a Dense net
        y = Dense(64, activation='relu')(sensor_input)
        y = Dense(32, activation='relu')(y)
    
    # Combine
    combined = Concatenate(name='concat')([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dense(1, name='yield_output')(z)
    
    model = Model(inputs=[ndvi_input, sensor_input], outputs=z)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model

# -----------------------------
# 4. Train & Evaluate
# -----------------------------
def train_and_evaluate(model, ndvi_data, sensor_data, yield_array, test_size=0.2, epochs=50, batch_size=16):
    # Split
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        ndvi_data, sensor_data, yield_array, test_size=test_size, random_state=42
    )
    
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early]
    )
    
    loss, mae = model.evaluate([X1_val, X2_val], y_val)
    print("Val loss:", loss, "Val MAE:", mae)
    return history

# -----------------------------
# 5. Fufll main
# -----------------------------
def main(ndvi_folder, sensor_folder, yield_csv_path):
    # 1. Load
    ndvi_data, sensor_data, yield_array = load_ndvi_sensor_yield(ndvi_folder, sensor_folder, yield_csv_path)
    
    # 2. Preprocess
    ndvi_data, sensor_data = preprocess_inputs(ndvi_data, sensor_data)
    print("NDVI data shape:", ndvi_data.shape)
    print("Sensor data shape:", sensor_data.shape)
    print("Yield shape:", yield_array.shape)
    
    # 3. Build model
    ndvi_shape = ndvi_data.shape[1:]  # e.g. (seq_len, H, W, channels)
    sensor_shape = sensor_data.shape[1:]  # e.g. (seq_len, num_features) or (num_features,)
    model = build_cnn_lstm_model(ndvi_shape, sensor_shape)
    
    # 4. Train & evaluate
    history = train_and_evaluate(model, ndvi_data, sensor_data, yield_array,
                                 test_size=0.2, epochs=100, batch_size=16)
    
    # Optionally, save the model
    model.save("cnn_lstm_yield_model.h5")
    print("Model saved as cnn_lstm_yield_model.h5")

if __name__ == "__main__":
    ndvi_folder = r"C:\shikher_jain\SIH\yeildModel\data\ndvi"
    sensor_folder = r"C:\shikher_jain\SIH\yeildModel\data\sensor"
    yield_csv = r"C:\shikher_jain\SIH\yeildModel\data\yield\Punjab&UP_Yield_2018To2021.csv"
    main(ndvi_folder, sensor_folder, yield_csv)
