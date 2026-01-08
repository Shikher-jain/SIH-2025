# 🌱 Crop Health Monitoring AI (SIH 2025)

This project is an **AI-powered crop health monitoring system** that uses **multispectral images (.tif)** and **environmental data** to predict crop status (Healthy/Unhealthy), compute vegetation indices (NDVI, SAVI, PRI), assess pest/disease risk factors, and provide actionable recommendations.

---

## 📂 Project Structure

```
final M2/
│── preprocess.py        # Image preprocessing (resize, normalize, NDVI calculation)
│── prepare_dataset.py   # Convert dataset into NumPy arrays (X_train.npy, y_train.npy)
│── train.py             # CNN model training and saving
│── predict.py           # Crop health prediction + risk factor calculation
│── visual.py            # Plot NDVI, SAVI, PRI using Plotly
│── requirements.txt     # Python dependencies
│── data/
│    ├── train/
│    │    ├── healthy/   # Healthy crop images (.tif)
│    │    ├── unhealthy/ # Unhealthy crop images (.tif)
│    └── sample_image.tif
```

---

## ⚙️ Setup Instructions

1. **Clone/Download this repo** to your local machine.

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**
   - Place your `.tif` images in the following structure:
     ```
     data/train/healthy/*.tif
     data/train/unhealthy/*.tif
     ```
   - Run preprocessing script:
     ```bash
     python prepare_dataset.py
     ```
   - ✅ Output: `X_train.npy`, `y_train.npy`

4. **Model Training**
   ```bash
   python train.py
   ```
   - ✅ Output: `crop_health_model.h5`

5. **Crop Health Prediction**
   - Update `predict.py` with:
     - Your `.tif` test image path
     - Latitude & longitude
     - API key for weather data
   - Run:
     ```bash
     python predict.py
     ```
   - ✅ Output: JSON result with:
     - Crop status (Healthy/Unhealthy)
     - Probability score
     - Vegetation indices (NDVI, SAVI, PRI)
     - Pest & disease risk factors
     - Recommendation

6. **Visualization**
   ```bash
   python visual.py
   ```
   - ✅ Output: Interactive Plotly charts for NDVI, SAVI, PRI.

---

## 🌍 Environmental Data (Weather API)

We use **OpenWeatherMap API** for:
- Humidity
- Temperature
- Wind speed
- Rainfall (can be used for soil moisture proxy)

👉 You must replace `API_KEY` in `predict.py` with your own OpenWeatherMap key.

---

## 📊 Vegetation Indices Used

- **NDVI**: `(NIR - Red) / (NIR + Red)`
- **SAVI**: `(1.5 * (NIR - Red)) / (NIR + Red + 0.5)`
- **PRI**: `(Green - Red) / (Green + Red)`

---

## 🚀 Workflow Summary

1. Prepare dataset → `prepare_dataset.py`
2. Train CNN model → `train.py`
3. Predict crop health → `predict.py`
4. Visualize vegetation indices → `visual.py`

---

## 📝 Notes

- `.tif` (GeoTIFF) images required for NDVI/SAVI/PRI (must include **NIR band**).  
- If only RGB images are available → conversion to `.tif` possible, but vegetation indices will be less accurate.  
- Add more training images for better accuracy.  
