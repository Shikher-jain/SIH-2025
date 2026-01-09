
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io

# FastAPI app instance
app = FastAPI()

# Load the pre-trained ML model
model = tf.keras.models.load_model('model.h5')

# Helper function to load .tif image
def load_tif_image(file: UploadFile):
    image_data = io.BytesIO(file.file.read())  # Read the image file
    image = Image.open(image_data)  # Open the image using PIL
    return np.array(image)  # Convert to NumPy array for model input

# Helper function to load .npy file
def load_npy_file(file: UploadFile):
    file_data = np.load(file.file)
    return file_data  # Return NumPy array

# Helper function to load CSV sensor data
def load_csv_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        return df.values  # Return as NumPy array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in processing CSV file: {str(e)}")

@app.post("/predict/")
async def predict(
    tif_file: UploadFile = File(...),
    npy_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
):
    # Load TIFF image
    try:
        image_data = load_tif_image(tif_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error in processing .tif file: " + str(e))
    
    # Load NumPy array
    try:
        npy_data = load_npy_file(npy_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error in processing .npy file: " + str(e))

    # Load sensor data from CSV
    try:
        sensor_data = load_csv_file(csv_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error in processing CSV file: " + str(e))

    # Combine image, NumPy array, and sensor data for model input
    # Here, flattening and concatenating the data as a simple example
    input_data = np.concatenate((image_data.flatten(), npy_data.flatten(), sensor_data.flatten()))
    input_data = input_data.reshape(1, -1)  # Reshape to fit model's input

    # Model prediction
    prediction = model.predict(input_data)

    return {"prediction": prediction.tolist()}


'''
to run
uvicorn app:app --reload

''' 