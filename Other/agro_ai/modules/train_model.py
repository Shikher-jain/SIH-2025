# modules/train_model.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from modules.model_cnn_lstm import build_fusion_model

ROOT = os.path.abspath(os.getcwd())
OUTPUT = os.path.join(ROOT, "outputs")
os.makedirs(os.path.join(OUTPUT, "models"), exist_ok=True)

def train_baseline_rf(csv_path=os.path.join("outputs","datasets","cell_features.csv")):
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=["int64","float64"]).copy()
    X = X.drop(columns=[c for c in X.columns if "weather_seq" in c], errors='ignore')
    y = df["label"].astype('category')
    y_codes = y.cat.codes.values
    X.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_codes, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(OUTPUT, "models", "rf_model.pkl"))
    # save label encoder mapping
    mapping = dict(enumerate(y.cat.categories))
    joblib.dump(mapping, os.path.join(OUTPUT, "models", "label_mapping.pkl"))
    print("[train] RandomForest trained and saved")
    return rf

def train_cnn_lstm(npz_path=os.path.join("outputs","images","images_and_seq.npz"), csv_path=os.path.join("outputs","datasets","cell_features.csv"), epochs=15):
    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]  # N,64,64,B
    seqs = data["seqs"]      # object array of 24x5
    seq_arr = np.stack([s for s in seqs])
    df = pd.read_csv(csv_path)
    tab_cols = [c for c in ["NDVI_mean","EVI_mean","SAVI_mean","SI_mean","BSI_mean"] if c in df.columns]
    X_tab = df[tab_cols].fillna(0).values
    y = df["label"].astype('category').cat.codes.values
    
    from sklearn.model_selection import train_test_split
    X_img_tr, X_img_te, X_seq_tr, X_seq_te, X_tab_tr, X_tab_te, y_tr, y_te = train_test_split(
        images, seq_arr, X_tab, y, test_size=0.2, random_state=42
    )
    input_shape_img = X_img_tr.shape[1:]
    input_shape_seq = X_seq_tr.shape[1:]
    input_shape_tab = X_tab_tr.shape[1:]
    model = build_fusion_model(input_shape_img, input_shape_seq, input_shape_tab, n_classes=len(np.unique(y)))


    from modules.validate_data import ensure_float32

    # Fix all inputs before training
    X_img_tr, X_seq_tr, X_tab_tr, y_tr = ensure_float32(X_img_tr, X_seq_tr, X_tab_tr, y_tr)
    X_img_te, X_seq_te, X_tab_te, y_te = ensure_float32(X_img_te, X_seq_te, X_tab_te, y_te)

    # Now train safely
    model.fit([X_img_tr, X_seq_tr, X_tab_tr], y_tr,
            validation_data=([X_img_te, X_seq_te, X_tab_te], y_te),
            epochs=10, batch_size=32)


    model.fit([X_img_tr, X_seq_tr, X_tab_tr], y_tr, validation_data=([X_img_te, X_seq_te, X_tab_te], y_te),epochs=epochs, batch_size=8)
    model.save(os.path.join(OUTPUT, "models", "cnn_lstm.h5"))
    print("[train] CNN+LSTM trained and saved")
    return model
