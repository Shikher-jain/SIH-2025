# modules/rolling_dataset.py
import os, json, numpy as np, pandas as pd
from modules.indices import compute_indices_from_image, resize_image
from modules.weather import parse_weather_json
import rasterio

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "datasets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

def read_tif(path):
    import rasterio
    try:
        with rasterio.open(path) as src:
            img = src.read().astype('float32')  # (bands, H, W)
            img = np.transpose(img, (1,2,0))   # HWC
            return img
    except Exception as e:
        print(f"[read_tif] {path}: {e}")
        return None

def make_future_weather_seq(weather_json, days=60, features_per_day=5):
    """
    Return a (days, features_per_day) numpy array.
    If weather_json has only 'seq24' or shorter, repeat/upsample to reach days.
    """
    # try structured 'seqN' or 'seq24' inside weather_json
    seq = []
    if isinstance(weather_json, dict):
        if "seq" in weather_json:
            base = weather_json["seq"]
        elif "seq24" in weather_json:
            base = weather_json["seq24"]
        elif "history" in weather_json:
            base = weather_json["history"]
        else:
            base = []
        # convert list of dicts to rows
        for s in base:
            seq.append([
                float(s.get("temperature", 28)),
                float(s.get("humidity", 50)),
                float(s.get("precipitation", 0)),
                float(s.get("windSpeed", 2)),
                float(s.get("pressure", 1005))
            ])
    # fallback: use current
    if not seq:
        cur = {
            "temperature": weather_json.get("current", {}).get("temperature", 28) if isinstance(weather_json, dict) else 28,
            "humidity": weather_json.get("current", {}).get("humidity", 50) if isinstance(weather_json, dict) else 50,
            "precipitation": weather_json.get("current", {}).get("precipitation", 0) if isinstance(weather_json, dict) else 0,
            "windSpeed": weather_json.get("current", {}).get("windSpeed", 2) if isinstance(weather_json, dict) else 2,
            "pressure": weather_json.get("current", {}).get("pressure", 1005) if isinstance(weather_json, dict) else 1005
        }
        seq = [ [cur["temperature"], cur["humidity"], cur["precipitation"], cur["windSpeed"], cur["pressure"]] ]

    seq = np.array(seq, dtype=np.float32)
    # repeat/tiling to reach required days
    if seq.shape[0] >= days:
        return seq[:days]
    # tile
    reps = int(np.ceil(days / seq.shape[0]))
    tiled = np.tile(seq, (reps,1))[:days]
    return tiled

def compute_window_label_from_image(img_array):
    """
    Rule-based label using multiple indices aggregated (same as dataset_builder)
    Returns 0/1/2 label.
    """
    inds = compute_indices_from_image(np.transpose(img_array, (2,0,1)))  # convert HWC->CHW inside indices
    weights = {"NDVI_mean":0.3,"EVI_mean":0.2,"GNDVI_mean":0.15,"NDWI_mean":0.1,"SAVI_mean":0.15,"MSI_mean":0.1}
    score = 0.0
    for k,w in weights.items():
        score += inds.get(k, 0.0) * w
    if score > 0.6: return 2
    elif score > 0.35: return 1
    else: return 0

def build_rolling_dataset(data_dir="data", future_days=60, window_count=4, window_len_days=15, image_target_size=(64,64)):
    """
    Build a dataset for predicting the next `window_count` windows (each window_len_days long).
    Assumptions:
      - If multiple satellite images per cell exist with date tags like *_window1.tif, *_window2.tif, ... they will be used
      - If only one satellite image exists, that single image will be used as 'current' image for all samples
      - If true future ground-truth labels are not available, rule-based labels (multi-index) from satellite snapshots will be used
    Returns:
      X_img: (N, H, W, C)
      X_weather_future: (N, future_days, 5)
      X_sensor: optional (N, num_sensor_features)  -- currently None unless sensor processed
      y_future_labels: (N, window_count) integers in {0,1,2}
      ids: list of cell ids
    """
    rows_meta = []
    X_img = []
    X_weather = []
    y_future = []
    ids = []

    for cell in sorted(os.listdir(data_dir)):
        cell_path = os.path.join(data_dir, cell)
        if not os.path.isdir(cell_path):
            continue
        try:
            # find satellite files for historical windows (if present)
            sat_files = sorted([f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and "satellite" in f.lower()])
            # choose a 'current' image: prefer one named 'current' or the latest file
            current_sat = None
            for f in sat_files[::-1]:
                if "current" in f.lower() or "latest" in f.lower():
                    current_sat = f; break
            if current_sat is None and sat_files:
                current_sat = sat_files[-1]
            if current_sat is None:
                print(f"[rolling_dataset] skipping {cell}: no satellite images")
                continue
            img = read_tif(os.path.join(cell_path, current_sat))
            if img is None:
                continue
            # resize for model
            img_resized = resize_image(img, target=image_target_size)  # CHW
            img_resized = np.transpose(img_resized, (1,2,0))  # CHW->HWC for model input

            # weather future sequence (try to read a future-weather file)
            weather_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".json") and "weather" in f.lower()), None)
            if weather_file:
                with open(os.path.join(cell_path, weather_file),'r') as f:
                    weather_json = json.load(f)
            else:
                weather_json = {}

            future_weather = make_future_weather_seq(weather_json, days=future_days)  # (days,5)

            # Build target labels for each future window:
            # Option A: If there are satellite files corresponding to future windows (e.g. *_w1.tif ...), use them
            # Option B: If not, compute rule-based labels by assuming availability of "future_satellite_windowX.tif"
            future_labels = []
            used_ground_truth = False
            for w in range(1, window_count+1):
                # try to find a file for this window
                # look for patterns like *_window{w}.tif or *_w{w}.tif or *_future{w}.tif
                target_candidates = [f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and (f"_window{w}" in f.lower() or f"_w{w}" in f.lower() or f"_future{w}" in f.lower())]
                if target_candidates:
                    target_img = read_tif(os.path.join(cell_path, target_candidates[0]))
                    if target_img is not None:
                        lbl = compute_window_label_from_image(target_img)
                        future_labels.append(lbl)
                        used_ground_truth = True
                        continue
                # fallback: if no future image, but multiple satellite images exist sorted by date, try to pick next one
                if len(sat_files) >= w+1:
                    candidate = sat_files[-(w+1)]  # older images as pseudo-future if naming unknown
                    cand_img = read_tif(os.path.join(cell_path, candidate))
                    if cand_img is not None:
                        future_labels.append(compute_window_label_from_image(cand_img))
                        used_ground_truth = True
                        continue
                # final fallback: use rule-based prediction by shifting current indices slightly (simulate deterioration/improvement)
                # compute indices of current and then apply small random perturbation to simulate future (not ideal but enables training)
                # better if user supplies real future images
                cur_inds = compute_indices_from_image(np.transpose(img, (2,0,1)))
                score = 0.0
                weights = {"NDVI_mean":0.3,"EVI_mean":0.2,"GNDVI_mean":0.15,"NDWI_mean":0.1,"SAVI_mean":0.15,"MSI_mean":0.1}
                for k,wgt in weights.items():
                    score += cur_inds.get(k,0.0) * wgt
                # simulate slight random drift
                drift = (-0.05) * w  # assume slight decline over time; you may change to +/-
                score_sim = score + drift
                if score_sim > 0.6: lab = 2
                elif score_sim > 0.35: lab = 1
                else: lab = 0
                future_labels.append(lab)

            # append sample
            X_img.append(img_resized.astype(np.float32))
            X_weather.append(future_weather.astype(np.float32))
            y_future.append(np.array(future_labels, dtype=np.int32))
            ids.append(cell)

        except Exception as e:
            print(f"[rolling_dataset] skipping {cell}: {e}")
            continue

    X_img = np.stack(X_img) if X_img else np.array([])
    X_weather = np.stack(X_weather) if X_weather else np.array([])
    y_future = np.stack(y_future) if y_future else np.array([])
    ids = np.array(ids)

    # Save to outputs
    np.savez_compressed(os.path.join(OUTPUT_DIR, "images", "rolling_dataset.npz"),
                        X_img=X_img, X_weather=X_weather, y_future=y_future, ids=ids)
    # Save meta csv
    df = pd.DataFrame({"cell_id": ids})
    df.to_csv(os.path.join(OUTPUT_DIR, "datasets", "rolling_cells.csv"), index=False)

    print(f"[rolling_dataset] Built: {X_img.shape[0]} samples, future windows={window_count}")
    return X_img, X_weather, y_future, ids
