"""
create_agro_project.py
Creates a complete AgroAI project folder and zips it.
Run: python create_agro_project.py
"""

import os, textwrap, json, zipfile

PROJECT = "AgroAI_Project"
MODULES_DIR = os.path.join(PROJECT, "modules")
DATA_DIR = os.path.join(PROJECT, "HectFarm-1")
OUTPUTS = os.path.join(PROJECT, "outputs")

os.makedirs(MODULES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS, "dashboards"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS, "predictions"), exist_ok=True)

# -------- requirements.txt --------
requirements = """\
numpy
pandas
rasterio
opencv-python
tensorflow
scikit-learn
matplotlib
seaborn
"""
with open(os.path.join(PROJECT, "requirements.txt"), "w") as f:
    f.write(requirements)

# -------- README.md --------
readme = textwrap.dedent(f"""
AgroAI_Project
=============
Auto-generated project skeleton for HectFarm-1 sample pipeline.

Folders:
 - HectFarm-1/           put your cell_x folders here (satellite/sensor/weather JSON)
 - modules/              python modules (data loader, indices, model, etc.)
 - outputs/              where generated images, dashboards, predictions are stored

How to use:
1. Put your real data in {DATA_DIR} as described in the project.
2. Create a python venv and install requirements:
   python -m venv venv
   venv\\Scripts\\activate    (Windows)
   source venv/bin/activate   (Linux/Mac)
   pip install -r requirements.txt
3. Run main:
   python main.py

Notes:
 - This skeleton implements a CNN embedding + RF classifier flow, JSON advisory reports and heatmap dashboards.
 - Edit paths in main.py if your data sits elsewhere.
""")
with open(os.path.join(PROJECT, "README.md"), "w") as f:
    f.write(readme)

# -------- modules content --------
files = {}

files["modules/indices.py"] = textwrap.dedent("""\
import numpy as np
import rasterio

def compute_indices_from_image(sat_file):
    with rasterio.open(sat_file) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] < 6:
        raise ValueError("Expected 6 bands")
    B, G, R, NIR, SWIR1, SWIR2 = arr[:6]
    eps = 1e-6
    indices = {}
    indices['NDVI']  = (NIR - R) / (NIR + R + eps)
    indices['EVI']   = 2.5*(NIR - R)/(NIR + 6*R - 7.5*B + 1 + eps)
    indices['SAVI']  = ((NIR - R)/(NIR + R + 0.5 + eps))*1.5
    indices['GNDVI'] = (NIR - G)/(NIR + G + eps)
    indices['NDWI']  = (G - NIR)/(G + NIR + eps)
    agg = {}
    for k, arrv in indices.items():
        agg[f"{k}_mean"] = float(np.nanmean(arrv))
        agg[f"{k}_std"]  = float(np.nanstd(arrv))
        agg[f"{k}_min"]  = float(np.nanmin(arrv))
        agg[f"{k}_max"]  = float(np.nanmax(arrv))
    return agg
""")

files["modules/dataset_builder.py"] = textwrap.dedent("""\
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

def build_dataset(root_dir=\"HectFarm-1\"):
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
                print(f\"[dataset] skip {cell}: no satellite tif\")
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
            print(f\"[❌] skip {cell}: {e}\")
    import pandas as pd
    df = pd.DataFrame(rows)
    if images:
        np.savez_compressed(OUTPUT_IMAGES, images=np.stack(images), seqs=np.array(seqs, dtype=object), ids=np.array(ids))
    df.to_csv(os.path.join('outputs','datasets','cell_features.csv'), index=False)
    print(f\"saved dataset rows={len(rows)}\")
    return df
""")

files["modules/model_utils.py"] = textwrap.dedent("""\
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cnn_embedding(img_shape=(64,64,6)):
    # expects HWC
    inputs = layers.Input(shape=img_shape, name='img_in')
    x = layers.Conv2D(16,3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32,3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    model = Model(inputs, x, name='cnn_embed')
    return model
""")

files["modules/report_generator.py"] = textwrap.dedent("""\
def generate_advisory_report(cell_id, label, indices_dict, weather_dict):
    label_map = {0: 'Stressed', 1: 'Moderate', 2: 'Healthy'}
    report = f\"Advisory Report for {cell_id}:\\n\"
    report += f\"1. Crop condition: {label_map.get(label, 'Unknown')}\\n\"
    # rules
    if indices_dict.get('NDVI_mean',0) < 0.35:
        report += \"2. Low vegetation detected — irrigation recommended.\\n\"
    if weather_dict.get('humidity',0) > 80:
        report += \"3. High humidity — monitor for fungal disease.\\n\"
    if weather_dict.get('precipitation',0) > 10:
        report += \"4. Heavy rain expected — delay spraying.\\n\"
    report += \"5. Monitor daily and log field observations.\\n\"
    return report
""")

files["modules/dashboard.py"] = textwrap.dedent("""\
import matplotlib.pyplot as plt
import seaborn as sns
import os
def generate_cell_dashboard(cell_id, label):
    os.makedirs('outputs/dashboards', exist_ok=True)
    path = os.path.join('outputs','dashboards', f'{cell_id}_dashboard.png')
    plt.figure(figsize=(6,2))
    sns.heatmap([[label]], annot=True, cmap='RdYlGn', cbar=True)
    plt.title(f'Cell {cell_id} Condition (0=Stressed,1=Moderate,2=Healthy)')
    plt.savefig(path)
    plt.close()
    return path
""")

files["modules/predictor.py"] = textwrap.dedent("""\
import numpy as np
def predict_crop(model_cnn, rf_model, images, seqs, ids):
    # images: CHW stacked array saved earlier -> convert to HWC for CNN
    # images shape (N, C, H, W)
    imgs_hwc = np.transpose(images, (0,2,3,1))
    embeddings = model_cnn.predict(imgs_hwc)
    # Flatten seqs into fixed-length features (if variable length, simple aggregation)
    seq_feats = np.array([s.mean(axis=0) for s in seqs])
    X_tab = np.hstack([seq_feats, embeddings])
    preds = rf_model.predict(X_tab)
    probs = rf_model.predict_proba(X_tab) if hasattr(rf_model, 'predict_proba') else None
    out = []
    for i, cid in enumerate(ids):
        out.append({
            'cell_id': cid,
            'label': int(preds[i]),
            'confidence': float(probs[i].max()) if probs is not None else None
        })
    return out
""")

files["main.py"] = textwrap.dedent("""\
import numpy as np
from modules.dataset_builder import build_dataset
from modules.model_utils import build_cnn_embedding
from modules.predictor import predict_crop
from modules.report_generator import generate_advisory_report
from modules.dashboard import generate_cell_dashboard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, os

# 1. BUILD dataset (from HectFarm-1/)
df = build_dataset()

data_np = np.load('outputs/images/images_and_seq.npz', allow_pickle=True)
images = data_np['images']        # (N, C, H, W)
seqs = data_np['seqs']            # (N, seq_len, feats)
ids = data_np['ids']

# 2. Prepare features
# Convert images to HWC for CNN
imgs_hwc = np.transpose(images, (0,2,3,1))
cnn = build_cnn_embedding(img_shape=imgs_hwc.shape[1:])  # HWC
embs = cnn.predict(imgs_hwc)

# seq features: mean over time
seq_feats = np.array([s.mean(axis=0) for s in seqs])

X = np.hstack([seq_feats, embs])

# Labels - if df has NDVI_mean use that as placeholder label strategy
labels = []
for idx, row in df.iterrows():
    ndvi = row.get('NDVI_mean', 0)
    if ndvi > 0.6: labels.append(2)
    elif ndvi > 0.35: labels.append(1)
    else: labels.append(0)
y = np.array(labels)

# 3. train/test
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X, y, ids, test_size=0.2, random_state=42)

# 4. Train RF classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)

# save models
os.makedirs('outputs/models', exist_ok=True)
joblib.dump(rf, 'outputs/models/rf_model.joblib')
cnn.save('outputs/models/cnn_embed.h5')

# 5. Predict on test set
pred_json = predict_crop(cnn, rf, images, seqs, ids)

# 6. Generate reports & dashboards
for item in pred_json:
    cid = item['cell_id']
    label = item['label']
    # fetch indices row if available
    row = df[df['cell_id'] == cid]
    indices_dict = row.iloc[0].to_dict() if not row.empty else {}
    # approximate weather summary from seqs stored earlier (find index)
    try:
        i = list(ids).index(cid)
        seq_arr = seqs[i]
        weather_summary = {'temperature': float(seq_arr.mean(axis=0)[0]),
                           'humidity': float(seq_arr.mean(axis=0)[1]),
                           'precipitation': float(seq_arr.mean(axis=0)[2]),
                           'windSpeed': float(seq_arr.mean(axis=0)[3]),
                           'pressure': float(seq_arr.mean(axis=0)[4])}
    except:
        weather_summary = {'temperature':28,'humidity':50,'precipitation':0,'windSpeed':2,'pressure':1005}
    report = generate_advisory_report(cid, label, indices_dict, weather_summary)
    # save report json
    outjson = {
        'cell_id': cid,
        'label': label,
        'confidence': item.get('confidence'),
        'report': report
    }
    os.makedirs('outputs/predictions', exist_ok=True)
    import json
    with open(f\"outputs/predictions/{cid}.json\", 'w') as fp:
        json.dump(outjson, fp, indent=2)
    # dashboard
    generate_cell_dashboard(cid, label)

print('Done. Outputs in outputs/ folder.')
""")

# write files
for rel, content in files.items():
    path = os.path.join(PROJECT, rel)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# write a small sample metadata file inside sample HectFarm-1 cell folder to guide user
sample_cell = os.path.join(DATA_DIR, "cell_1_1")
os.makedirs(sample_cell, exist_ok=True)
meta_sample = {
    "cell_id": "cell_1_1",
    "lat": 30.9, "lon": 75.8,
    "notes": "Place your satellite/sensor .tif and weather .json files in this folder."
}
with open(os.path.join(sample_cell, "cell_1_1_metadata.json"), "w") as f:
    json.dump(meta_sample, f, indent=2)

# zip the project
zipname = PROJECT + ".zip"
with zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files_in in os.walk(PROJECT):
        for file in files_in:
            full = os.path.join(root, file)
            arcname = os.path.relpath(full, os.path.dirname(PROJECT))
            zf.write(full, arcname)

print(f"Project folder '{PROJECT}' created and zipped as '{zipname}'.")
print("Next: create venv, install requirements and place your real HectFarm-1 data into the HectFarm-1/ folder.")
