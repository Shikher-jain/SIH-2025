# main.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from modules.feature_extractor import read_tif, resize_image_hwc, compute_indices_from_hwc, compute_health_score, label_from_score
from modules.model import build_cnn_lstm_model

# ---------- User configurable ----------
DATA_DIR = "data"   # folder with cell_xxx subfolders
OUTPUT_DIR = "outputs"
IMG_TARGET = (64,64)
SEQ_LEN = 24        # LSTM timesteps expected
SEQ_FEATURES = 5    # temperature, humidity, precipitation, windSpeed, pressure
EPOCHS = 15
BATCH = 8
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "datasets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

def prepare_sequence_from_weather_json(weather_json, n_steps=SEQ_LEN):
    """
    weather_json may contain 'seq24' (list of dicts) or a list under 'history' or be a simple dict 'current'.
    We return an array (n_steps, 5). If insufficient length, we tile/repeat last value or pad with current.
    """
    rows = []
    if isinstance(weather_json, str):
        # path given
        try:
            with open(weather_json, 'r') as f:
                data = json.load(f)
        except:
            data = {}
    else:
        data = weather_json or {}

    seq_src = []
    if "seq24" in data and isinstance(data["seq24"], list):
        seq_src = data["seq24"]
    elif "history" in data and isinstance(data["history"], list):
        seq_src = data["history"]
    elif isinstance(data, list):
        seq_src = data
    else:
        # try current
        cur = data.get("current", {})
        seq_src = [ {
            "temperature": cur.get("temperature", cur.get("main", {}).get("temp", 28)),
            "humidity": cur.get("humidity", cur.get("main", {}).get("humidity", 50)),
            "precipitation": cur.get("precipitation", 0),
            "windSpeed": cur.get("windSpeed", cur.get("wind", {}).get("speed", 2)),
            "pressure": cur.get("pressure", cur.get("main", {}).get("pressure", 1005))
        } ]

    # normalize seq_src to list of dicts with required keys
    seq_list = []
    for item in seq_src:
        if isinstance(item, dict):
            seq_list.append([
                float(item.get("temperature", item.get("temp", 28))),
                float(item.get("humidity",  item.get("hum", 50))),
                float(item.get("precipitation", item.get("precip", 0))),
                float(item.get("windSpeed", item.get("wind", {}).get("speed", 2))),
                float(item.get("pressure", item.get("pressure", 1005)))
            ])
        else:
            # if item is list/tuple
            vals = list(item)
            needed = vals[:5] + [0]*(5 - len(vals))
            seq_list.append([float(v) for v in needed])

    seq_arr = np.array(seq_list, dtype=np.float32)
    if seq_arr.shape[0] == 0:
        # fallback: use current defaults
        seq_arr = np.tile(np.array([28,50,0,2,1005], dtype=np.float32), (n_steps,1))
    elif seq_arr.shape[0] >= n_steps:
        seq_arr = seq_arr[:n_steps]
    else:
        # tile/repeat last row to fill to n_steps
        reps = int(np.ceil(n_steps / seq_arr.shape[0]))
        seq_arr = np.tile(seq_arr, (reps,1))[:n_steps]
    return seq_arr

# ---------- Build dataset ----------
rows = []
images = []
seqs = []
ids = []

for cell in sorted(os.listdir(DATA_DIR)):
    cell_path = os.path.join(DATA_DIR, cell)
    if not os.path.isdir(cell_path):
        continue

    # find satellite .tif (prefer contains 'satellite')
    sat_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".tif") and "satellite" in f.lower()), None)
    if sat_file is None:
        # fallback: any tif
        sat_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".tif")), None)
    if sat_file is None:
        print(f"[skip] {cell}: no tif")
        continue
    sat_path = os.path.join(cell_path, sat_file)
    img_hwc = read_tif(sat_path)
    if img_hwc is None:
        print(f"[skip] {cell}: could not read tif")
        continue

    # weather json
    weather_file = next((f for f in os.listdir(cell_path) if f.lower().endswith(".json") and "weather" in f.lower()), None)
    weather_json = {}
    if weather_file:
        try:
            with open(os.path.join(cell_path, weather_file),'r') as wf:
                weather_json = json.load(wf)
        except Exception as e:
            print(f"[warn] {cell} weather json read error: {e}")
            weather_json = {}

    # compute indices and label
    try:
        inds = compute_indices_from_hwc(img_hwc)
    except Exception as e:
        print(f"[warn] {cell} compute indices error: {e}")
        continue

    score = compute_health_score(inds)
    label = label_from_score(score)

    # prepare image resized HWC
    img_resized = resize_image_hwc(img_hwc, target=IMG_TARGET)   # HWC

    # prepare sequence (SEQ_LEN x 5)
    seq_arr = prepare_sequence = None
    try:
        seq_arr = prepare_sequence_from_weather_json(weather_json, n_steps=SEQ_LEN)
    except Exception as e:
        print(f"[warn] {cell} prepare seq error: {e}")
        seq_arr = np.tile(np.array([28,50,0,2,1005],dtype=np.float32), (SEQ_LEN,1))

    rows.append({
        "cell_id": cell,
        "sat_file": sat_file,
        "label": label,
        **inds,
        "score": score
    })
    images.append(img_resized.astype(np.float32))
    seqs.append(seq_arr.astype(np.float32))
    ids.append(cell)

# save tabular
df = pd.DataFrame(rows)
csv_out = os.path.join(OUTPUT_DIR, "datasets", "cell_features.csv")
df.to_csv(csv_out, index=False)
print(f"[saved] {csv_out} rows={len(df)}")

# stack arrays
X_img = np.stack(images)          # (N,H,W,C)
X_seq = np.stack(seqs)           # (N,SEQ_LEN,5)
y = np.array(df['label'].values, dtype=np.int32)
cell_ids = np.array(ids)

print("Shapes:", X_img.shape, X_seq.shape, y.shape)

# ---------- Train/test split ----------
X_img_train, X_img_test, X_seq_train, X_seq_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X_img, X_seq, y, cell_ids, test_size=0.2, random_state=42, stratify=y
)

# ---------- Build & compile model ----------
model = build_cnn_lstm_model(img_shape=X_img.shape[1:], seq_len=SEQ_LEN, seq_features=SEQ_FEATURES, n_classes=3)
model.summary()

# ---------- Training with safeguards ----------
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
ckpt_path = os.path.join(OUTPUT_DIR, "models")
os.makedirs(ckpt_path, exist_ok=True)
ckpt_file = os.path.join(ckpt_path, "best_model.h5")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(ckpt_file, monitor="val_loss", save_best_only=True, save_weights_only=False)
]

history = model.fit(
    [X_img_train, X_seq_train], y_train,
    validation_data=([X_img_test, X_seq_test], y_test),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=callbacks
)

# ---------- Predict & evaluate ----------
pred_probs = model.predict([X_img_test, X_seq_test])
pred_labels = np.argmax(pred_probs, axis=1)

print("\nClassification report:")
print(classification_report(y_test, pred_labels, digits=4))

# save confusion matrix
cm = confusion_matrix(y_test, pred_labels)
print("Confusion matrix:\n", cm)

# ---------- Generate simple advisory reports for test set ----------
for i, cid in enumerate(ids_test):
    pred = int(pred_labels[i])
    row = df[df['cell_id'] == cid].iloc[0].to_dict()
    # compute simple weather summary
    w_mean = X_seq_test[i].mean(axis=0)
    weather_summary = {
        "temperature": float(w_mean[0]),
        "humidity": float(w_mean[1]),
        "precipitation": float(w_mean[2]),
        "windSpeed": float(w_mean[3]),
        "pressure": float(w_mean[4])
    }
    # build a quick report text
    from modules.report_generator import generate_advisory_report
    report_text = generate_advisory_report(cid, pred, row, weather_summary)
    # save report
    with open(os.path.join(OUTPUT_DIR, "reports", f"{cid}_report.txt"), 'w', encoding='utf-8') as rf:
        rf.write(report_text)

print("[done] Reports written to", os.path.join(OUTPUT_DIR, "reports"))

# ---------- Plot heatmap of predicted labels for test set ----------
labels_matrix = np.array([pred_labels])
plt.figure(figsize=(12,2))
sns.heatmap(labels_matrix, cmap='RdYlGn', cbar=True, annot=True)
plt.title("Predicted crop label (0=Stressed,1=Moderate,2=Healthy) across test cells")
plt.yticks([]); plt.xlabel("Test sample index")
plt.show()
