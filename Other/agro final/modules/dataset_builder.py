# modules/dataset_builder.py
import os
import json
import numpy as np
import pandas as pd
from modules.indices import compute_indices_from_image, resize_image
from modules.weather import parse_weather_json

ROOT = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(os.path.join(OUTPUT_DIR, "datasets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)


def read_tif(path):
    """Read a .tif file as a numpy array (HWC format)."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            img = src.read()  # (bands, H, W)
            img = np.transpose(img, (1, 2, 0))  # HWC
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
            # Satellite file
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
                continue

            # Sensor file (optional)
            sensor_file = next(
                (f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and "sensor" in f.lower()), None
            )
            sensor_path = os.path.join(cell_path, sensor_file) if sensor_file else None

            # Metadata JSON
            meta_file = next((f for f in os.listdir(cell_path) if f.endswith("_metadata.json")), None)
            meta = {}
            if meta_file:
                try:
                    with open(os.path.join(cell_path, meta_file), 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[⚠️ meta json error] {cell_folder}: {e}")

            lat = meta.get("lat") or meta.get("latitude") or 30.9
            lon = meta.get("lon") or meta.get("longitude") or 75.8
            crop_type = meta.get("crop_type", 0)

            # Weather JSON (optional)
            weather_file = next((f for f in os.listdir(cell_path) if f.endswith("_weather.json")), None)
            if weather_file:
                try:
                    with open(os.path.join(cell_path, weather_file), 'r', encoding='utf-8') as f:
                        weather_json = json.load(f)
                    weather = parse_weather_json(weather_json)
                except Exception as e:
                    print(f"[⚠️ weather json error] {cell_folder}: {e}")
                    weather = parse_weather_json({})
            else:
                weather = parse_weather_json({})

            seq_arr = np.array(weather["seq24"], dtype=np.float32)

            # Compute indices from image
            inds = compute_indices_from_image(img)

            # Resize image for CNN
            img_resized = resize_image(img, target=(64, 64))

            # Simple NDVI-based labeling
            ndvi = inds.get("NDVI_mean", 0)
            if ndvi > 0.6:
                label = 2  # Healthy
            elif ndvi > 0.35:
                label = 1  # Moderate
            else:
                label = 0  # Stressed

            row = {
                "cell_id": cell_folder,
                "lat": float(lat),
                "lon": float(lon),
                "crop_type": crop_type,
                "label": label,
                **inds,
                "temperature": weather["current"]["temperature"],
                "humidity": weather["current"]["humidity"],
                "pressure": weather["current"]["pressure"],
                "windSpeed": weather["current"]["windSpeed"],
                "precipitation": weather["current"]["precipitation"],
                "weather_seq": seq_arr.tolist()
            }

            rows.append(row)
            images.append(img_resized.astype(np.float32))
            seqs.append(seq_arr)
            ids.append(cell_folder)

        except Exception as e:
            print(f"[❌ ERROR] Skipping {cell_folder}: {e}")
            continue

    # Save tabular dataset
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Save images & sequences
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
    return df
