import json
import numpy as np

def parse_weather(weather_file):
    if isinstance(weather_file, str):
        with open(weather_file, 'r', encoding='utf-8') as f:   # <-- encoding specified
            data = json.load(f)
    else:
        data = weather_file

    seq24 = []
    for i in range(24):
        seq24.append([
            data.get("temperature",28),
            data.get("humidity",50),
            data.get("precipitation",0),
            data.get("windSpeed",2),
            data.get("pressure",1005)
        ])
    seq_arr = np.array(seq24, dtype=np.float32)
    return seq_arr

