import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, TimeDistributed, Conv2D, MaxPooling2D,
                                     Flatten, LSTM, Dense, Concatenate, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping

def load_and_resize_file(filepath, target_shape):
    """
    Load .npy file and resize spatial dimensions (height, width) to target_shape.
    Assumes input shape is (H, W) or (H, W, C).
    """
    data = np.load(filepath)
    if data.ndim == 2:
        # grayscale: (H, W)
        data_resized = cv2.resize(data, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        return data_resized
    elif data.ndim == 3:
        # (H, W, C)
        channels = data.shape[2]
        resized_channels = []
        for c in range(channels):
            resized_c = cv2.resize(data[:, :, c], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized_c)
        return np.stack(resized_channels, axis=-1)
    else:
        raise ValueError(f"Unsupported data shape for resizing: {data.shape}")

def load_and_reshape_seq(ndvi_folder, sensor_folder, yield_csv_path, target_height=316, target_width=316):
    df_yield = pd.read_csv(yield_csv_path)
    yield_array = df_yield['yield'].values
    num_samples = len(yield_array)

    # List and sort files
    ndvi_files = sorted([f for f in os.listdir(ndvi_folder) if f.endswith('.npy')])
    sensor_files = sorted([f for f in os.listdir(sensor_folder) if f.endswith('.npy')])

    print(f"Number of yield samples: {num_samples}")
    print(f"Number of NDVI files: {len(ndvi_files)}")
    print(f"Number of Sensor files: {len(sensor_files)}")

    # Auto-detect time_steps
    if len(ndvi_files) % num_samples == 0 and len(sensor_files) % num_samples == 0:
        time_steps_ndvi = len(ndvi_files) // num_samples
        time_steps_sensor = len(sensor_files) // num_samples
        if time_steps_ndvi != time_steps_sensor:
            raise ValueError(f"Mismatch in time steps for NDVI ({time_steps_ndvi}) and sensor ({time_steps_sensor}) files")
        time_steps = time_steps_ndvi
        print(f"Auto-detected time_steps: {time_steps}")
    else:
        raise ValueError("Files count not divisible by number of yield samples.")

    expected_ndvi_files = num_samples * time_steps
    expected_sensor_files = num_samples * time_steps

    if len(ndvi_files) != expected_ndvi_files or len(sensor_files) != expected_sensor_files:
        raise ValueError("Number of files does not match expected based on time steps and samples.")

    # Load and resize NDVI data into array: (num_samples, time_steps, H, W, 1)
    ndvi_data = np.zeros((num_samples, time_steps, target_height, target_width, 1), dtype=np.float32)
    sensor_channels = None

    # Load and resize sensor data into array: (num_samples, time_steps, H, W, sensor_channels)
    # First load one sensor file to detect channels and shape
    sample_sensor = np.load(os.path.join(sensor_folder, sensor_files[0]))
    sensor_channels = sample_sensor.shape[2] if sample_sensor.ndim == 3 else 1
    sensor_data = np.zeros((num_samples, time_steps, target_height, target_width, sensor_channels), dtype=np.float32)

    for i in range(num_samples):
        for t in range(time_steps):
            ndvi_fp = os.path.join(ndvi_folder, ndvi_files[i * time_steps + t])
            sensor_fp = os.path.join(sensor_folder, sensor_files[i * time_steps + t])

            # Load and resize NDVI (assumed grayscale)
            ndvi_resized = load_and_resize_file(ndvi_fp, (target_height, target_width))
            if ndvi_resized.ndim == 2:
                ndvi_resized = ndvi_resized[..., np.newaxis]  # add channel dim

            ndvi_data[i, t] = ndvi_resized.astype(np.float32)

            # Load and resize sensor data
            sensor_loaded = np.load(sensor_fp)
            if sensor_loaded.ndim == 2:
                # (H, W) single channel
                sensor_loaded = sensor_loaded[..., np.newaxis]
            elif sensor_loaded.ndim == 3:
                pass
            else:
                raise ValueError(f"Unexpected sensor data shape: {sensor_loaded.shape}")

            sensor_resized_channels = []
            for c in range(sensor_loaded.shape[2]):
                resized_c = cv2.resize(sensor_loaded[:, :, c], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                sensor_resized_channels.append(resized_c)
            sensor_resized = np.stack(sensor_resized_channels, axis=-1)
            sensor_data[i, t] = sensor_resized.astype(np.float32)

    return ndvi_data, sensor_data, yield_array

def preprocess_inputs(ndvi_data, sensor_data):
    # Normalize NDVI to [0, 1]
    ndvi_data = ndvi_data / 255.0
    # Standardize sensor data (flatten samples and time)
    N, T, H, W, C = sensor_data.shape
    sensor_reshaped = sensor_data.reshape((N * T * H * W, C))
    scaler = StandardScaler()
    sensor_scaled = scaler.fit_transform(sensor_reshaped)
    sensor_scaled = sensor_scaled.reshape((N, T, H, W, C))
    return ndvi_data, sensor_scaled

def build_cnn_lstm_model(ndvi_shape, sensor_shape):
    ndvi_input = Input(shape=ndvi_shape, name='ndvi_input')  # (T, H, W, C)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(ndvi_input)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)  # better than flatten for spatial dims
    x = LSTM(64)(x)

    sensor_input = Input(shape=sensor_shape, name='sensor_input')  # (T, H, W, C)
    y = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(sensor_input)
    y = TimeDistributed(MaxPooling2D((2, 2)))(y)
    y = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(y)
    y = TimeDistributed(MaxPooling2D((2, 2)))(y)
    y = TimeDistributed(GlobalAveragePooling2D())(y)
    y = LSTM(64)(y)

    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dense(1, name='yield_output')(z)

    model = Model(inputs=[ndvi_input, sensor_input], outputs=z)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model

def train_and_evaluate(model, ndvi_data, sensor_data, yield_array, test_size=0.2, epochs=50, batch_size=16):
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        ndvi_data, sensor_data, yield_array, test_size=test_size, random_state=42)

    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early]
    )

    loss, mae = model.evaluate([X1_val, X2_val], y_val)
    print(f"Validation loss: {loss:.4f}, MAE: {mae:.4f}")
    return history

def main(ndvi_folder, sensor_folder, yield_csv_path):
    target_height = 316
    target_width = 316

    # Load and reshape sequences (auto time_steps)
    ndvi_data, sensor_data, yield_array = load_and_reshape_seq(ndvi_folder, sensor_folder, yield_csv_path,
                                                               target_height=target_height,
                                                               target_width=target_width)

    # Preprocess inputs
    ndvi_data, sensor_data = preprocess_inputs(ndvi_data, sensor_data)

    print(f"NDVI data shape: {ndvi_data.shape}")
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Yield shape: {yield_array.shape}")

    # Build model
    ndvi_shape = ndvi_data.shape[1:]  # (T, H, W, C)
    sensor_shape = sensor_data.shape[1:]
    model = build_cnn_lstm_model(ndvi_shape, sensor_shape)

    # Train and evaluate
    train_and_evaluate(model, ndvi_data, sensor_data, yield_array)


if __name__ == "__main__":
    ndvi_folder = r"data\ndvi"
    sensor_folder = r"data\sensor"
    yield_csv = r"data\yield\Punjab&UP_Yield_2018To2021.csv"
    main(ndvi_folder, sensor_folder, yield_csv)
