import os, json, numpy as np
import rasterio
import cv2
from modules.indices import compute_indices_from_image

OUTPUT_IMAGES = "outputs/images/images_and_seq.npz"

def read_tif_as_hwc(path):
    try:
        with rasterio.open(path) as src:
            img = src.read().astype('float32')  # (bands,H,W)
            img = img.transpose(1,2,0)         # HWC
            return img
    except Exception as e:
        print(f"[read_tif] {path}: {e}")
        return None

def resize_hwc(img_hwc, target=(64,64)):
    h,w,c = img_hwc.shape
    bands = []
    for b in range(c):
        band = img_hwc[:,:,b]
        band_r = cv2.resize(band, target, interpolation=cv2.INTER_LINEAR)
        bands.append(band_r)
    resized = np.stack(bands, axis=-1)
    return resized.astype('float32')

def build_dataset(root_dir="HectFarm-1"):
    rows = []
    images = []
    seqs = []
    ids = []
    for cell in sorted(os.listdir(root_dir)):
        cell_path = os.path.join(root_dir, cell)
        if not os.path.isdir(cell_path):
            continue
        try:
            sat_file = next((f for f in os.listdir(cell_path) if 'satellite' in f.lower()), None)
            if not sat_file:
                print(f"[dataset] skip {cell}: no satellite tif")
                continue
            sat_path = os.path.join(cell_path, sat_file)
            inds = compute_indices_from_image(sat_path)
            # sensor tif (optional)
            sensor_file = next((f for f in os.listdir(cell_path) if 'sensor' in f.lower()), None)
            sensor_img = None
            if sensor_file:
                sensor_img = read_tif_as_hwc(os.path.join(cell_path, sensor_file))
            # weather json
            weather_file = next((f for f in os.listdir(cell_path) if 'weather' in f.lower()), None)
            weather_json = {}
            if weather_file:
                with open(os.path.join(cell_path, weather_file), 'r') as wf:
                    weather_json = json.load(wf)
            seq24 = weather_json.get('seq24', [])
            seq_array = []
            for s in seq24:
                seq_array.append([
                    s.get('temperature', 28),
                    s.get('humidity', 50),
                    s.get('precipitation', 0),
                    s.get('windSpeed', 2),
                    s.get('pressure', 1005)
                ])
            if len(seq_array) == 0:
                # fallback fill
                seq_array = [[28,50,0,2,1005]] * 24
            seq_array = np.array(seq_array, dtype='float32')
            rows.append({'cell_id': cell, **inds})
            # prepare image: use sensor if present else use satellite bands stacked/resized
            if sensor_img is not None:
                img_use = resize_hwc(sensor_img, target=(64,64))
            else:
                sat_hwc = read_tif_as_hwc(sat_path)
                img_use = resize_hwc(sat_hwc, target=(64,64))
            images.append(img_use.transpose(2,0,1))  # CHW for CNN builder later
            seqs.append(seq_array)
            ids.append(cell)
        except Exception as e:
            print(f"[❌] skip {cell}: {e}")
    import pandas as pd
    df = pd.DataFrame(rows)
    if images:
        np.savez_compressed(OUTPUT_IMAGES, images=np.stack(images), seqs=np.array(seqs, dtype=object), ids=np.array(ids))
    df.to_csv(os.path.join('outputs','datasets','cell_features.csv'), index=False)
    print(f"saved dataset rows={len(rows)}")
    return df
