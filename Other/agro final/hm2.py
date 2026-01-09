import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1️⃣ Load satellite image (GeoTIFF)
sat_path = r'C:\shikher_jain\SIH\agro final\data\cell_1_2\test-1_satellite_2025-09-18_2025-09-23T13-55-47-106Z.tif'
src = rasterio.open(sat_path)

# Assume bands: 3=Red, 4=NIR (adjust as per your image)
red = src.read(3).astype(float)
nir = src.read(4).astype(float)

# Calculate NDVI
ndvi = (nir - red) / (nir + red + 1e-6)

# 2️⃣ Load yield data (CSV with lat, lon, yield, district, state)
yield_df = pd.read_csv(r'C:\shikher_jain\SIH\agro final\ICRISAT-District Level Data.csv\ICRISAT-District Level Data.csv')  # columns: lat, lon, yield, district
coord = pd.read_csv(r'C:\shikher_jain\SIH\agro final\ICRISAT-District Level Data.csv\Atal Jal 31 March 2021 .xlsx - Sheet1.csv')  # columns: lat, lon, district, state

# 3️⃣ Merge yield_df and coord on lat, lon (or district if lat/lon not unique)
merged = pd.merge(yield_df, coord, on=['lat', 'lon'], how='left')
# If lat/lon not unique, use district as key:
# merged = pd.merge(yield_df, coord, on='district', how='left')

# 4️⃣ Convert yield points to raster indices
merged['row'], merged['col'] = zip(*[src.index(lon, lat) for lon, lat in zip(merged.lon, merged.lat)])

# 5️⃣ Extract NDVI values at yield points
merged['ndvi'] = [ndvi[row, col] if 0 <= row < ndvi.shape[0] and 0 <= col < ndvi.shape[1] else np.nan for row, col in zip(merged.row, merged.col)]
merged.dropna(subset=['ndvi'], inplace=True)

# 6️⃣ Prepare training data
X = merged[['ndvi']].values
y = merged['yield'].values

# 7️⃣ Train Random Forest Regression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 8️⃣ Predict yield for full image based on NDVI
ndvi_flat = ndvi.flatten().reshape(-1, 1)
pred_yield_flat = model.predict(ndvi_flat)
pred_yield = pred_yield_flat.reshape(ndvi.shape)

# 9️⃣ Assign predicted yield to each point in merged
merged['pred_yield'] = model.predict(merged[['ndvi']].values)

# 🔟 State and district-wise yield aggregation
district_yield = merged.groupby(['state', 'district'])['pred_yield'].mean().reset_index()
print("State & District-wise Predicted Yield:")
print(district_yield)

# 11️⃣ Plot predicted yield heatmap
plt.figure(figsize=(10,8))
plt.imshow(pred_yield, cmap='YlOrRd')
plt.colorbar(label='Predicted Yield')
plt.title('Predicted Yield Heatmap')
plt.axis('off')
plt.show()