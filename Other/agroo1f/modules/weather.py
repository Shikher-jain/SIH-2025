import numpy as np
import json

def parse_weather_json(weather_json):
    """
    Returns current weather and sequence features (last 24 hours)
    """
    seq24 = weather_json.get("seq24", [])
    seq_arr = np.array([
        [
            s.get("temperature",28),
            s.get("humidity",50),
            s.get("precipitation",0),
            s.get("windSpeed",2),
            s.get("pressure",1005)
        ]
        for s in seq24
    ], dtype=np.float32)

    current = weather_json.get("current", {})
    features = {
        "temperature": current.get("temperature", 28),
        "humidity": current.get("humidity", 50),
        "pressure": current.get("pressure", 1005),
        "windSpeed": current.get("windSpeed", 2),
        "precipitation": current.get("precipitation", 0),
        "seq24": seq_arr
    }
    return features
