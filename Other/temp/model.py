import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import cv2

# 1. FUNCTION: Load a single .npy file
def load_npy_file(file_path):
    """
    Loads a single .npy file.

    Args:
        file_path (str): Path to the .npy file.
    
    Returns:
        np.ndarray: Loaded data from the .npy file.
    """
    print(f"Loading file: {file_path}")
    return np.load(file_path, allow_pickle=True)

# 2. FUNCTION: Data Preprocessing
def preprocess_data(data):
    """
    Preprocesses the loaded data for training:
    - Normalize NDVI images.
    - Standardize sensor data.
    
    Args:
        data (np.ndarray): The loaded data array.
    
    Returns:
        tuple: Processed NDVI images, sensor data, and yield data.
    """
    # Separate the data into NDVI images, sensor data, and yield
    ndvi_images = data[:, 0]
    sensor_data = data[:, 1]
    yield_data = data[:, 2]
    
    # Normalize NDVI images
    ndvi_images = np.array([img / 255.0 for img in ndvi_images])
    ndvi_images = np.expand_dims(ndvi_images, axis=-1)  # Add channel dimension

    # Standardize sensor data
    scaler = StandardScaler()
    sensor_data = scaler.fit_transform(sensor_data)
    
    return ndvi_images, sensor_data, yield_data

# 3. FUNCTION: Build Model
def build_model(ndvi_input_shape, sensor_input_shape):
    """
    Builds the Keras model for yield prediction.
    
    Args:
        ndvi_input_shape (tuple): Shape of the NDVI image input (height, width, channels).
        sensor_input_shape (int): Number of features in the sensor data.
    
    Returns:
        keras.Model: Compiled Keras model.
    """
    # NDVI Image Input and CNN Feature Extraction (using simple Flatten for simplicity)
    image_input = Input(shape=ndvi_input_shape)
    x = Flatten()(image_input)

    # Sensor Data Input and Dense Layers
    sensor_input = Input(shape=(sensor_input_shape,))
    y = Dense(64, activation='relu')(sensor_input)
    y = Dense(32, activation='relu')(y)

    # Combine both image and sensor features
    combined = Concatenate()([x, y])

    # Output Layer for Yield Prediction (single value)
    output = Dense(1)(combined)

    # Build and compile the model
    model = Model(inputs=[image_input, sensor_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# 4. FUNCTION: Train Model
def train_model(model, ndvi_images, sensor_data, yield_data, epochs=100, batch_size=32, validation_split=0.2):
    """
    Trains the model.
    
    Args:
        model (keras.Model): The compiled Keras model.
        ndvi_images (np.ndarray): Processed NDVI images.
        sensor_data (np.ndarray): Processed sensor data.
        yield_data (np.ndarray): Actual yield data.
        epochs (int, optional): Number of epochs. Default is 100.
        batch_size (int, optional): Batch size. Default is 32.
        validation_split (float, optional): Fraction of the data to use for validation. Default is 0.2.
    
    Returns:
        keras.callbacks.History: History object containing training logs.
    """
    # EarlyStopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        [ndvi_images, sensor_data],  # Inputs: NDVI images and sensor data
        yield_data,                  # Output: Yield data
        epochs=epochs,               # Number of epochs
        batch_size=batch_size,       # Batch size
        validation_split=validation_split,  # Validation split
        callbacks=[early_stopping]   # Apply EarlyStopping callback
    )
    
    return history

# 5. FUNCTION: Evaluate Model
def evaluate_model(model, ndvi_images, sensor_data, yield_data):
    """
    Evaluates the model on the dataset.
    
    Args:
        model (keras.Model): The trained model.
        ndvi_images (np.ndarray): Processed NDVI images.
        sensor_data (np.ndarray): Processed sensor data.
        yield_data (np.ndarray): Actual yield data.
    
    Returns:
        float: Mean Absolute Error (MAE).
    """
    loss, mae = model.evaluate([ndvi_images, sensor_data], yield_data)
    print(f'Mean Absolute Error on Full Dataset: {mae}')
    return mae

# 6. FUNCTION: Make Predictions
def make_predictions(model, ndvi_images, sensor_data):
    """
    Makes predictions using the trained model.
    
    Args:
        model (keras.Model): The trained model.
        ndvi_images (np.ndarray): Processed NDVI images.
        sensor_data (np.ndarray): Processed sensor data.
    
    Returns:
        np.ndarray: Model predictions.
    """
    predictions = model.predict([ndvi_images, sensor_data])
    return predictions

# 7. MAIN FUNCTION: Full pipeline for loading, training, and evaluating
def main(input_folder, epochs=100, batch_size=32):
    """
    Main function to run the full pipeline: load data, preprocess, build model, train, evaluate, and predict for all .npy files in a folder.
    
    Args:
        input_folder (str): Path to the folder containing .npy files.
        epochs (int, optional): Number of epochs for training. Default is 100.
        batch_size (int, optional): Batch size for training. Default is 32.
    """
    # Get all .npy files from the input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    if not npy_files:
        print("No .npy files found in the folder.")
        return
    
    # Loop over each .npy file and process
    for npy_file in npy_files:
        file_path = os.path.join(input_folder, npy_file)
        print(f"\nProcessing file: {npy_file}")

        # 1. Load the data
        data = load_npy_file(file_path)
        
        # 2. Preprocess the data
        ndvi_images, sensor_data, yield_data = preprocess_data(data)
        
        # 3. Build the model
        model = build_model(ndvi_input_shape=(315, 316, 1), sensor_input_shape=sensor_data.shape[1])
        
        # 4. Train the model
        history = train_model(model, ndvi_images, sensor_data, yield_data, epochs=epochs, batch_size=batch_size)
        
        # 5. Evaluate the model
        evaluate_model(model, ndvi_images, sensor_data, yield_data)
        
        # 6. Make predictions
        predictions = make_predictions(model, ndvi_images, sensor_data)
        print(f"First 5 predictions for {npy_file}: {predictions[:5]}")  # Display first 5 predictions

if __name__ == "__main__":
    # Set the path to the folder containing .npy files
    input_folder = 'data/Input'  # Update this with the actual path to your folder containing .npy files
    
    # Run the full pipeline
    main(input_folder, epochs=100, batch_size=32)
    