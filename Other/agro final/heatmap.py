# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt

# sat_path = r'C:\shikher_jain\SIH\agro final\data\cell_1_2\test-1_satellite_2025-09-18_2025-09-23T13-55-47-106Z.tif'
# src = rasterio.open(sat_path)

# height, width = src.height, src.width

# pred_yield = np.full((height, width), 4.0)  # fixed value everywhere

# plt.figure(figsize=(10, 8))
# plt.imshow(pred_yield, cmap='YlOrRd')
# plt.colorbar(label='Yield (tons/ha)')
# plt.title('Predicted Yield Heatmap (Fixed 4 tons/ha)')
# plt.axis('off')
# plt.show()

import rasterio
import numpy as np
import plotly.express as px

# ------------------------
# 1️⃣ Load satellite image
# ------------------------
sat_path = r'C:\shikher_jain\SIH\agro final\data\cell_1_2\test-1_satellite_2025-09-18_2025-09-23T13-55-47-106Z.tif'
src = rasterio.open(sat_path)

height, width = src.height, src.width
print(f"Image size: {height} x {width}")

# ------------------------
# 2️⃣ Create fixed yield array
# ------------------------
base_yield = 4.0
pred_yield = np.full((height, width), base_yield)

# ------------------------
# 3️⃣ Apply weights (optional)
# Example: center higher, edges lower
# ------------------------
y_indices, x_indices = np.indices((height, width))
center_y, center_x = height / 2, width / 2
distance = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
max_distance = np.max(distance)
weight = 1.5 - (distance / max_distance)  # center ~1.5, edges ~0.5
pred_yield_weighted = pred_yield * weight

# ------------------------
# 4️⃣ Downsample if too large
# ------------------------
max_size = 1000  # max pixels along one axis for interactive plotting
factor_y = max(1, height // max_size)
factor_x = max(1, width // max_size)
pred_yield_small = pred_yield_weighted[::factor_y, ::factor_x]
print(f"Downsampled size: {pred_yield_small.shape}")

# ------------------------
# 5️⃣ Plot interactive heatmap with Plotly
# ------------------------
fig = px.imshow(pred_yield_small,
                color_continuous_scale='YlOrRd',
                labels={'color': 'Yield (tons/ha)'},
                title='Predicted Yield Heatmap (Weighted, tons/ha)')

# Remove axis ticks
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

fig.show()
