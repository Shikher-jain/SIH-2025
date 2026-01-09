import numpy as np
import matplotlib.pyplot as plt

arr = np.load(r"C:\shikher_jain\SIH\model\data\data\Agra11.npy")

# List of fields to visualize
fields_to_plot = ['NDVI', 'ECe_mean', 'N_mean', 'P_mean', 'pH_mean', 'OC_mean']

plt.figure(figsize=(15, 10))
for i, field in enumerate(fields_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.imshow(arr[field], cmap="viridis")
    plt.title(field)
    plt.axis("off")
plt.tight_layout()
plt.savefig("soil_vs_ndvi.png", dpi=300)
plt.show()
