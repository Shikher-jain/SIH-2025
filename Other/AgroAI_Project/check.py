import numpy as np
import matplotlib.pyplot as plt

data_np = np.load('outputs/images/images_and_seq.npz', allow_pickle=True)
images = data_np['images']  # Shape: (N, C, H, W)

# Select the first image and transpose to (H, W, C)
img = images[0].transpose(1, 2, 0)  # CHW to HWC

# If more than 3 channels, select first 3 for RGB
if img.shape[-1] > 3:
    img = img[:, :, :3]

plt.imshow(img)
plt.show()
