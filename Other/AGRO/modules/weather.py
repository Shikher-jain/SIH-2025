# modules/weather.py
import json

def parse_weather_json(json_file):
    """
    Return dict with current weather + last 24 hour sequence.
    """
    try:
        if isinstance(json_file, str):
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = json_file
    except:
        data = {}

    current = {
        "temperature": data.get("current", {}).get("main", {}).get("temp", 28),
        "humidity": data.get("current", {}).get("main", {}).get("humidity", 60),
        "pressure": data.get("current", {}).get("main", {}).get("pressure", 1005),
        "windSpeed": data.get("current", {}).get("wind", {}).get("speed", 2),
        "precipitation": data.get("current", {}).get("precipitation", 0)
    }

    seq24 = []
    for i in range(24):
        seq24.append({
            "temperature": current["temperature"],
            "humidity": current["humidity"],
            "precipitation": current["precipitation"],
            "windSpeed": current["windSpeed"],
            "pressure": current["pressure"]
        })

    return {"current": current, "seq24": seq24}

def parse_weather_json(weather_json):
    """
    Parse weather json and return dictionary and sequence features
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
