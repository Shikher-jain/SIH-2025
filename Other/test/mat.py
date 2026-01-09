import numpy as np
import scipy.io
import os
from PIL import Image

# === Step 1: Load the .mat file ===
mat_path = 'C:\shikher_jain\SIH\Indian_pines_corrected (2).mat'  # Replace with your file
data = scipy.io.loadmat(mat_path)

# === Step 2: Inspect available variables ===
print("Available variables in .mat file:")
for key in data:
    if not key.startswith('__'):
        print(f" - {key}: shape = {np.shape(data[key])}")

# === Step 3: Choose the right variable ===
# The hyperspectral data is stored in 'indian_pines'
hyperspectral_data = data['indian_pines']  # shape = (145, 145, 220) - height, width, spectral_bands
print(f"Hyperspectral data shape: {hyperspectral_data.shape}")
print(f"Data type: {hyperspectral_data.dtype}")
print(f"Value range: {hyperspectral_data.min()} to {hyperspectral_data.max()}")

# === Optional: Create output directory ===
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# === Step 4: Extract and save dataset ===
# Save the complete hyperspectral dataset as numpy array
np.save('indian_pines_dataset.npy', hyperspectral_data)
print(f"Saved complete dataset to indian_pines_dataset.npy")

# === Step 5: Create RGB visualization from selected bands ===
# Select bands for RGB visualization (you can adjust these)
red_band = hyperspectral_data[:, :, 50]    # Band 50 for red
green_band = hyperspectral_data[:, :, 100] # Band 100 for green
blue_band = hyperspectral_data[:, :, 150]  # Band 150 for blue

# Normalize each band to 0-255
def normalize_band(band):
    band_min, band_max = band.min(), band.max()
    if band_max > band_min:
        return ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
    else:
        return np.zeros_like(band, dtype=np.uint8)

red_norm = normalize_band(red_band)
green_norm = normalize_band(green_band)
blue_norm = normalize_band(blue_band)

# Create RGB image
rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
rgb_pil = Image.fromarray(rgb_image, mode='RGB')
rgb_pil.save(os.path.join(output_dir, 'indian_pines_rgb.png'))
print(f"Saved RGB visualization to {output_dir}/indian_pines_rgb.png")

# === Step 6: Save individual spectral bands (first 10 as examples) ===
for i in range(min(10, hyperspectral_data.shape[2])):
    band = hyperspectral_data[:, :, i]
    band_norm = normalize_band(band)
    band_pil = Image.fromarray(band_norm, mode='L')
    band_pil.save(os.path.join(output_dir, f'band_{i:03d}.png'))

print(f"Saved first 10 spectral bands to {output_dir}/")

# === Step 7: Save dataset information ===
with open('dataset_info.txt', 'w') as f:
    f.write(f"Indian Pines Hyperspectral Dataset\n")
    f.write(f"================================\n")
    f.write(f"Shape: {hyperspectral_data.shape}\n")
    f.write(f"Spatial dimensions: {hyperspectral_data.shape[0]} x {hyperspectral_data.shape[1]}\n")
    f.write(f"Spectral bands: {hyperspectral_data.shape[2]}\n")
    f.write(f"Data type: {hyperspectral_data.dtype}\n")
    f.write(f"Value range: {hyperspectral_data.min()} to {hyperspectral_data.max()}\n")
    f.write(f"\nFiles created:\n")
    f.write(f"- indian_pines_dataset.npy: Complete dataset\n")
    f.write(f"- indian_pines_rgb.png: RGB visualization\n")
    f.write(f"- band_000.png to band_009.png: First 10 spectral bands\n")

print("Dataset extraction completed!")
