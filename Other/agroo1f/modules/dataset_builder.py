import os
import json
import numpy as np
import pandas as pd
import rasterio
from modules.indices import compute_indices_from_image, resize_image
from modules.weather import parse_weather_json

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,"datasets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,"images"), exist_ok=True)

def read_tif(path):
    try:
        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img, (1,2,0))  # HWC
            return img
    except:
        return None

def build_dataset(base_path="data"):
    rows, images, seqs, ids = [], [], [], []

    for cell_folder in sorted(os.listdir(base_path)):
        cell_path = os.path.join(base_path, cell_folder)
        if not os.path.isdir(cell_path): continue

        # Satellite file
        sat_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and "satellite" in f.lower()), None)
        if not sat_file: continue
        img = read_tif(os.path.join(cell_path,sat_file))
        if img is None: continue

        # Weather file
        weather_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".json") and "weather" in f.lower()), None)
        if weather_file:
            with open(os.path.join(cell_path,weather_file),'r', encoding='utf-8') as f:
                weather_json = json.load(f)
        else:
            weather_json = {}

        w = parse_weather_json(weather_json)
        seq_arr = w['seq24']

        # Indices
        inds = compute_indices_from_image(img)

        # Resize image for CNN
        img_resized = resize_image(img,(64,64))

        # Compute multi-index label
        score = 0
        weights = {"NDVI_mean":0.3,"EVI_mean":0.2,"GNDVI_mean":0.15,"NDWI_mean":0.1,"SAVI_mean":0.15,"MSI_mean":0.1}
        for k,wgt in weights.items():
            score += inds.get(k,0)*wgt
        if score>0.6: label=2
        elif score>0.35: label=1
        else: label=0

        row = {"cell_id":cell_folder, "label":label, **inds,
               "temperature":w["temperature"], "humidity":w["humidity"],
               "pressure":w["pressure"],
               "windSpeed":w["windSpeed"], "precipitation":w["precipitation"]}
        rows.append(row)
        images.append(img_resized.astype(np.float32))
        seqs.append(np.array(seq_arr,dtype=np.float32))
        ids.append(cell_folder)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR,"datasets","cell_features.csv"), index=False)
    np.savez_compressed(os.path.join(OUTPUT_DIR,"images","images_and_seq.npz"),
                        images=np.stack(images) if images else np.array([]),
                        seqs=np.array(seqs,dtype=object),
                        ids=np.array(ids))
    return df
