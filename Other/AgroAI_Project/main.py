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
    with open(f"outputs/predictions/{cid}.json", 'w') as fp:
        json.dump(outjson, fp, indent=2)
    # dashboard
    generate_cell_dashboard(cid, label)

print('Done. Outputs in outputs/ folder.')
