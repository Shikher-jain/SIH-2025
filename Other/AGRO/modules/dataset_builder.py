# modules/dataset_builder.py
import os, json, numpy as np, pandas as pd
from modules.indices import compute_indices_from_image, resize_image
from modules.weather import parse_weather_json

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(os.path.join(OUTPUT_DIR, "datasets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

def read_tif(path):
    import rasterio
    try:
        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img, (1,2,0))
            return img
    except Exception as e:
        print(f"[⚠️ read_tif] Skipping {path}: {e}")
        return None

def build_dataset(data_dir="data"):
    rows, images, seqs, ids = [], [], [], []

    for cell_folder in sorted(os.listdir(data_dir)):
        cell_path = os.path.join(data_dir, cell_folder)
        if not os.path.isdir(cell_path): continue

        try:
            sat_file = next((f for f in os.listdir(cell_path) if "satellite" in f.lower() and f.endswith(".tif")), None)
            if not sat_file: continue
            img = read_tif(os.path.join(cell_path, sat_file))
            if img is None: continue

            meta_file = next((f for f in os.listdir(cell_path) if f.endswith("_metadata.json")), None)
            meta = json.load(open(os.path.join(cell_path, meta_file))) if meta_file else {}
            crop_type = meta.get("crop_type", 0)
            lat = float(meta.get("lat", 30.9))
            lon = float(meta.get("lon", 75.8))

            weather_file = next((f for f in os.listdir(cell_path) if f.endswith("_weather.json")), None)
            w = parse_weather_json(os.path.join(cell_path, weather_file)) if weather_file else parse_weather_json({})

            inds = compute_indices_from_image(os.path.join(cell_path, sat_file))
            img_resized = resize_image(img)
            images.append(img_resized.astype(np.float32))
            seq_arr = np.array([[s["temperature"], s["humidity"], s["precipitation"], s["windSpeed"], s["pressure"]] for s in w["seq24"]])
            seqs.append(seq_arr)
            ids.append(cell_folder)

            ndvi_mean = inds.get("NDVI_mean",0)
            label = 2 if ndvi_mean>0.6 else 1 if ndvi_mean>0.35 else 0

            row = {"cell_id":cell_folder, "lat":lat,"lon":lon,"crop_type":crop_type,"label":label, **inds,
                   "temperature":w["current"]["temperature"], "humidity":w["current"]["humidity"],
                   "pressure":w["current"]["pressure"], "windSpeed":w["current"]["windSpeed"],
                   "precipitation":w["current"]["precipitation"]}
            rows.append(row)

        except Exception as e:
            print(f"[❌ ERROR] Skipping {cell_folder}: {e}")
            continue

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR,"datasets","cell_features.csv")
    df.to_csv(out_csv,index=False)

    np.savez_compressed(os.path.join(OUTPUT_DIR,"images","images_and_seq.npz"),
                        images=np.stack(images), ids=np.array(ids), seqs=np.array(seqs, dtype=object))

    print(f"[dataset] saved {len(rows)} rows -> {out_csv}")
    return df, images, ids
