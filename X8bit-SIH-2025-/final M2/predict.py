import tensorflow as tf
import numpy as np
from preprocess import preprocess_image
import requests

# ------------------ Load Model ------------------ #
model = tf.keras.models.load_model("crop_health_model.h5")

# ------------------ Crop Prediction & Risk ------------------ #
def calculate_risk_factors(ndvi, savi, pri, env):
    pest_risk = 0.8 if env["humidity"] > 70 and 25 <= env["temperature"] <= 35 and env["leaf_wetness"] > 0.2 else 0.3

    disease_risk = 0
    if ndvi < 0.4 or savi < 0.3:
        disease_risk += 0.6
    if env["soil_moisture"] < 0.15 or env["soil_moisture"] > 0.35:
        disease_risk += 0.3

    return {
        "pest_risk": round(min(1.0, pest_risk), 2),
        "disease_risk": round(min(1.0, disease_risk), 2)
    }

def predict_crop_status(img_path, lat, lon, env):
    img, ndvi = preprocess_image(img_path)

    # Model prediction
    X = np.expand_dims(img[:, :, :3] / 255.0, axis=0)
    probs = model.predict(X)[0]
    label = np.argmax(probs)
    status = "Healthy" if label == 0 else "Unhealthy"

    # Vegetation indices
    savi = (1.5 * (img[:, :, 3] - img[:, :, 0])) / (img[:, :, 3] + img[:, :, 0] + 0.5)
    pri = (img[:, :, 1] - img[:, :, 0]) / (img[:, :, 1] + img[:, :, 0] + 1e-6)

    return {
        "latitude": lat,
        "longitude": lon,
        "crop_health": {"status": status, "probability_score": float(np.max(probs))},
        "vegetation_indices": {"NDVI": float(np.mean(ndvi)), "SAVI": float(np.mean(savi)), "PRI": float(np.mean(pri))},
        "risk_factors": calculate_risk_factors(np.mean(ndvi), np.mean(savi), np.mean(pri), env),
        "recommendation": "Apply pesticide" if status == "Unhealthy" else "No action needed"
    }

# ------------------ Leaf Wetness Estimation ------------------ #
def estimate_leaf_wetness(humidity, temp, rainfall=0):
    if rainfall > 0:
        return 1.0
    elif humidity > 90 and 15 <= temp <= 30:
        return 0.7
    elif humidity > 80:
        return 0.4
    else:
        return 0.1

# ------------------ Fetch Environment Data ------------------ #
def fetch_openweathermap_env(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        env_data = {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "wind_speed": data['wind']['speed'],
            "leaf_wetness": estimate_leaf_wetness(data['main']['humidity'], data['main']['temp']),
            "soil_moisture": 0.20
        }
        return env_data
    else:
        print("OpenWeatherMap Error:", data)
        return None
def fetch_tomorrowio_env(lat, lon, api_key):
    url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{lat},{lon}",
        "apikey": api_key,
        "fields": "temperature,humidity,windSpeed,precipitationIntensity",
        "timesteps": "1h",
        "timezone": "Asia/Kolkata"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        interval = data['data']['timelines'][0]['intervals'][0]
        values = interval['values']
        env_data = {
            "temperature": values['temperature'],
            "humidity": values['humidity'],
            "wind_speed": values['windSpeed'],
            "leaf_wetness": estimate_leaf_wetness(values['humidity'], values['temperature'], values.get('precipitationIntensity', 0)),
            "soil_moisture": 0.20
        }
        return env_data
    else:
        print("Tomorrow.io Error:", data)
        return None

# ------------------ Example Usage ------------------ #
lat, lon = 28.6139, 77.2090
OPENWEATHER_API_KEY = "b3b6ce66c807e5519c12592db541db9d"
TOMORROW_API_KEY = "RxUc1q8QfRCKtJodaHooSA4Zw04hZviX"

# Fetch environment data
env_open = fetch_openweathermap_env(lat, lon, OPENWEATHER_API_KEY)
env_tomorrow = fetch_tomorrowio_env(lat, lon, TOMORROW_API_KEY)

# Predict crop status using Tomorrow.io data as example
if env_tomorrow:
    result = predict_crop_status("image.tif", lat, lon, env_tomorrow)
    print(result)

# Optional: Also print OpenWeatherMap data
if env_open:
    print("\nOpenWeatherMap Environment Data:")
    print(env_open)
