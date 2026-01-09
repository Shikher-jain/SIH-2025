# modules/dataset_builder.py
import os
import json
import numpy as np
import pandas as pd
import rasterio
from modules.indices import compute_indices_from_image, resize_image
from modules.weather import parse_weather_json

ROOT = os.path.abspath(os.path.join(os.getcwd()))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "datasets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

def read_tif(path):
    try:
        with rasterio.open(path) as src:
            img = src.read()  # (bands, height, width)
            img = np.transpose(img, (1, 2, 0))  # HWC format
            return img
    except Exception as e:
        print(f"[⚠️ read_tif] Skipping {path}: {e}")
        return None


def build_dataset(base_path=DATA_DIR, out_csv=None):
    if out_csv is None:
        out_csv = os.path.join(OUTPUT_DIR, "datasets", "cell_features.csv")

    rows = []
    images = []
    seqs = []
    ids = []

    for cell_folder in sorted(os.listdir(base_path)):
        cell_path = os.path.join(base_path, cell_folder)
        if not os.path.isdir(cell_path):
            continue

        try:
            # satellite tif (6-band)
            sat_file = next(
                (f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and "satellite" in f.lower()),
                None
            )
            if not sat_file:
                print(f"[dataset] skipping {cell_folder}: no satellite tif found")
                continue
            sat_path = os.path.join(cell_path, sat_file)
            img = read_tif(sat_path)
            if img is None:
                continue  # corrupted image skip

            # sensor tif optional (ignored for now in tabular)
            sensor_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and "sensor" in f.lower()), None)

            # metadata json
            meta = {}
            meta_file = next((f for f in os.listdir(cell_path) if f.endswith("_metadata.json")), None)
            if meta_file:
                try:
                    with open(os.path.join(cell_path, meta_file), 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[⚠️ meta json error] {cell_folder}: {e}")
                    meta = {}

            lat = meta.get("lat") or meta.get("latitude") or 30.9
            lon = meta.get("lon") or meta.get("longitude") or 75.8
            # crop_type = meta.get("crop_type", "unknown")
            crop_type = meta.get("crop_type", 0)

            # weather json (per cell) optional
            weather_file = next((f for f in os.listdir(cell_path) if f.endswith("_weather.json")), None)
            if weather_file:
                try:
                    with open(os.path.join(cell_path, weather_file), 'r', encoding='utf-8') as f:
                        weather_json = json.load(f)
                    w = parse_weather_json(weather_json)
                except Exception as e:
                    print(f"[⚠️ weather json error] {cell_folder}: {e}")
                    mock = {"current": {"main": {"temp": 28, "humidity": 60, "pressure": 1005},
                                        "wind": {"speed": 2}, "clouds": {"all": 0}},
                            "daily": [{"precipitation": 0}]}
                    w = parse_weather_json(mock)
            else:
                mock = {"current": {"main": {"temp": 28, "humidity": 60, "pressure": 1005},
                                    "wind": {"speed": 2}, "clouds": {"all": 0}},
                        "daily": [{"precipitation": 0}]}
                w = parse_weather_json(mock)

            seq24 = w["seq24"]
            seq_arr = np.array([
                [s["temperature"], s["humidity"], s["precipitation"], s["windSpeed"], s["pressure"]]
                for s in seq24
            ])

            # compute indices (on raw)
            inds = compute_indices_from_image(img)

            # resize image for CNN
            img_resized = resize_image(img, target=(64, 64))

            # simple rule label (replace later with manual labels)
            ndvi = inds.get("NDVI_mean", 0)
            if ndvi > 0.6:
                # label = "Healthy"
                label = 2
            elif ndvi > 0.35:
                # label = "Moderate"
                label = 1
            else:
                # label = "Stressed"
                label = 0

            row = {
                "cell_id": cell_folder,
                "lat": float(lat),
                "lon": float(lon),
                "crop_type": crop_type,
                "label": label,
                **inds,
                "temperature": w["current"]["temperature"],
                "humidity": w["current"]["humidity"],
                "pressure": w["current"]["pressure"],
                "windSpeed": w["current"]["windSpeed"],
                "precipitation": w["current"]["precipitation"],
                "weather_seq": seq_arr.tolist()
            }
            rows.append(row)
            images.append(img_resized.astype(np.float32))
            seqs.append(seq_arr)
            ids.append(cell_folder)

        except Exception as e:
            print(f"[❌ ERROR] Skipping {cell_folder}: {e}")
            continue

    # save tabular
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # save images/sequences
    try:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "images", "images_and_seq.npz"),
            images=np.stack(images) if images else np.array([]),
            ids=np.array(ids),
            seqs=np.array(seqs, dtype=object)
        )
    except Exception as e:
        print(f"[⚠️ save error] images_and_seq.npz: {e}")

    print(f"[dataset] saved {len(rows)} rows -> {out_csv}")
    return out_csv
