import numpy as np

def get_npy_shape(file_path):
    try:
        arr = np.load(file_path)
        return arr.shape
    except Exception as e:
        return f"Error: {e}"

# Example usage
shape = get_npy_shape("./test18_output/Tarn_Taran/1/feature_1_stacked.npy")
print("Array Shape:", shape)