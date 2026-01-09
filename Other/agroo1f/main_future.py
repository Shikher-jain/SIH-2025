# main_future.py
import numpy as np
from modules.rolling_dataset import build_rolling_dataset
from modules.model_future import build_future_predictor
from modules.report_generator import generate_advisory_report  # we reuse earlier generator per window
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Build rolling dataset (assumes data/ folder present)
X_img, X_weather, y_future, ids = build_rolling_dataset(data_dir="data", future_days=60, window_count=4, window_len_days=15, image_target_size=(64,64))

if X_img.size == 0:
    raise RuntimeError("No samples built. Check data/ directory and satellite/weather files.")

# Convert shapes:
# X_img expected shape: (N, H, W, C)
# X_weather: (N, future_days, 5)
# y_future: (N, window_count)
N = X_img.shape[0]
window_count = y_future.shape[1]
n_classes = 3

# 2) Train/test split (stratify by first-window label to preserve distribution)
first_labels = y_future[:,0]
train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=42, stratify=first_labels)

X_img_tr = X_img[train_idx]; X_img_te = X_img[test_idx]
X_weather_tr = X_weather[train_idx]; X_weather_te = X_weather[test_idx]
y_tr = y_future[train_idx]; y_te = y_future[test_idx]
ids_tr = ids[train_idx]; ids_te = ids[test_idx]

# 3) Build model
model = build_future_predictor(img_shape=X_img.shape[1:], future_days=X_weather.shape[1], seq_features=X_weather.shape[2], window_count=window_count, n_classes=n_classes)
model.summary()

# 4) Train model
# Keras expects y with shape (batch, window_count) for sparse_categorical_crossentropy.
history = model.fit([X_img_tr, X_weather_tr], y_tr, validation_data=([X_img_te, X_weather_te], y_te),
                    epochs=30, batch_size=8, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# 5) Predict on test set
pred_probs = model.predict([X_img_te, X_weather_te])  # shape (M, window_count, n_classes)
pred_labels = np.argmax(pred_probs, axis=-1)          # shape (M, window_count)

# 6) For each test cell, generate per-window advisory reports
all_reports = {}
for i, cid in enumerate(ids_te):
    preds_for_cell = pred_labels[i]  # array length window_count
    # prepare indices and a representative weather summary (we can use mean of future weather)
    indices_dict = {}  # we can compute indices from X_img_te[i] if needed (we'll leave it to dataset CSV)
    weather_dict = {
        "temperature": float(X_weather_te[i][:,0].mean()),
        "humidity": float(X_weather_te[i][:,1].mean()),
        "precipitation": float(X_weather_te[i][:,2].mean()),
        "windSpeed": float(X_weather_te[i][:,3].mean()),
        "pressure": float(X_weather_te[i][:,4].mean())
    }
    window_reports = []
    for w, label in enumerate(preds_for_cell, start=1):
        # generate advisory for this window
        window_id = f"{cid}_window{w}"
        # if you have indices for the cell in CSV, you can pass them; here we pass empty placeholder
        rep = generate_advisory_report(window_id, int(label), {}, weather_dict)
        window_reports.append({"window": w, "label": int(label), "report": rep})
    all_reports[cid] = window_reports

# 7) Print sample reports
for cid, reports in list(all_reports.items())[:5]:
    print("="*40)
    print(f"Reports for cell: {cid}")
    for r in reports:
        print(f"Window {r['window']}: label={r['label']}\n{r['report']}\n")

# 8) Visualize aggregated horizon: for test set, heatmap of labels across windows
labels_grid = np.stack([all_reports[cid][i]['label'] for cid in ids_te for i in range(window_count)])
# reshape to (window_count, num_cells) for heatmap
labels_matrix = np.stack([all_reports[cid][i]['label'] for i in range(window_count)] for cid in ids_te).T  # careful shape
labels_matrix = np.array([[all_reports[cid][i]['label'] for cid in ids_te] for i in range(window_count)])
plt.figure(figsize=(max(8, labels_matrix.shape[1]/2), 4))
sns.heatmap(labels_matrix, annot=True, fmt="d", cmap='RdYlGn', yticklabels=[f"W{i+1}" for i in range(window_count)], xticklabels=ids_te)
plt.title("Predicted labels per future window (rows=window index, cols=cells)")
plt.show()
