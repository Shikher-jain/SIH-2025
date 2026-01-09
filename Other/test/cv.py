import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load heatmap array
array = np.load("Agra1_mask.npy")


# Extract channels
red = array['Red']
green = array['Green']
blue = array['Blue']

# Stack into (H, W, 3)
rgb_image = np.stack([red, green, blue], axis=-1)
# Normalize the array to 0–255
rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)
rgb_image = rgb_image.astype(np.uint8)

# Apply a heatmap colormap using OpenCV
heatmap = cv2.applyColorMap(rgb_image, cv2.COLORMAP_JET)

# Save heatmap image
cv2.imwrite("heatmap_output.png", heatmap)
print("Saved as heatmap_output.png")


# Plot heatmap
plt.imshow(heatmap)
plt.show()

'''
NDVI maps: Red → dense vegetation, Blue → no vegetation
Soil moisture: Red → wettest, Blue → driest
Elevation: Red → high elevation, Blue → low
Temperature: Red → hot, Blue → cold
Model attention maps / saliency: Red → most focus, Blue → none
'''