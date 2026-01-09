# modules/weather.py
import json
from datetime import datetime, timedelta
import math

def parse_weather_json(w):
    """
    Accepts dictionary (as in your sample). Returns dict:
      - current: {temperature, humidity, pressure, windSpeed, precipitation}
      - seq24: list of 24 hourly dicts (temperature, humidity, precipitation, windSpeed, pressure, timestamp)
    """
    if isinstance(w, str):
        with open(w, 'r', encoding='utf-8') as f:
            w = json.load(f)

    # Extract current safely
    cur = w.get("current", {})
    main = cur.get("main", {}) if isinstance(cur, dict) else {}
    wind = cur.get("wind", {}) if isinstance(cur, dict) else {}
    clouds = cur.get("clouds", {}) if isinstance(cur, dict) else {}

    temp = float(main.get("temp", 0.0))
    humidity = float(main.get("humidity", 0.0))
    pressure = float(main.get("pressure", 0.0))
    wind_speed = float(wind.get("speed", 0.0))
    precip = float(0.0)
    # daily aggregated precipitation may be available in w["daily"][0]["precipitation"]
    daily = w.get("daily", [])
    if daily and isinstance(daily, list):
        precip = float(daily[0].get("precipitation", 0.0))

    current = {
        "temperature": temp,
        "humidity": humidity,
        "pressure": pressure,
        "windSpeed": wind_speed,
        "precipitation": precip
    }

    # Build 24-hour synthetic hourly sequence using daily averages if present,
    # otherwise repeat current with a diurnal sinusoidal variation.
    seq = []
    for h in range(24):
        # diurnal temp variation - simple sine around avg temperature
        hourly_temp = temp + 3.0 * math.sin(2 * math.pi * (h / 24.0) - math.pi/2)
        hourly_hum = max(0.0, min(100.0, humidity + 5.0 * math.cos(2 * math.pi * (h / 24.0))))
        seq.append({
            "timestamp": (datetime.utcnow() + timedelta(hours=h)).isoformat(),
            "temperature": float(round(hourly_temp, 2)),
            "humidity": float(round(hourly_hum, 2)),
            "precipitation": float(precip / 24.0),  # distribute daily precip evenly
            "windSpeed": float(wind_speed),
            "pressure": float(pressure)
        })

    return {"current": current, "seq24": seq}
