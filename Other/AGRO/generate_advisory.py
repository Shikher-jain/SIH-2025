# generate_advisory.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
IMAGES_FILE = os.path.join(OUTPUT_DIR, "images", "images_and_seq.npz")
CSV_FILE    = os.path.join(OUTPUT_DIR, "datasets", "cell_features.csv")
MODEL_FILE  = os.path.join(OUTPUT_DIR, "models", "cnn_lstm_model.h5")

def generate_advisory():
    model = load_model(MODEL_FILE)

    npz = np.load(IMAGES_FILE, allow_pickle=True)
    images = npz['images']
    seqs   = npz['seqs']
    ids    = npz['ids']

    df = pd.read_csv(CSV_FILE)

    # preprocess
    images_scaled = images.astype(np.float32)
    for b in range(images_scaled.shape[1]):
        band = images_scaled[:, b, :, :]
        images_scaled[:, b, :, :] = (band - np.mean(band)) / (np.std(band)+1e-6)

    seqs_scaled = np.array(seqs, dtype=np.float32)
    for i in range(seqs_scaled.shape[2]):
        col = seqs_scaled[:,:,i]
        seqs_scaled[:,:,i] = (col - np.mean(col)+1e-6) / (np.std(col)+1e-6)

    images_scaled = np.transpose(images_scaled, (0,2,3,1))

    # predict
    preds = model.predict([images_scaled, seqs_scaled])
    labels_pred = np.argmax(preds, axis=1)
    df['pred_label'] = labels_pred

    label_map = {0:"Stressed",1:"Moderate",2:"Healthy"}
    df['pred_label_str'] = df['pred_label'].map(label_map)

    # Save advisory txt
    for idx, row in df.iterrows():
        cell = row['cell_id']
        advisory = f"Advisory Report for {cell}:\nCrop condition: {row['pred_label_str']}"
        report_file = os.path.join(OUTPUT_DIR, f"{cell}_advisory.txt")
        with open(report_file,"w") as f:
            f.write(advisory)
        print(f"✅ Saved advisory: {report_file}")

    # Heatmap
    heatmap_array = np.zeros((1,len(df)))
    for i, lbl in enumerate(labels_pred):
        heatmap_array[0,i] = lbl

    plt.figure(figsize=(12,2))
    import seaborn as sns
    sns.heatmap(heatmap_array, annot=True, fmt="d", cmap="YlGnBu", xticklabels=df['cell_id'], yticklabels=["Label"])
    heatmap_file = os.path.join(OUTPUT_DIR,"predicted_labels_heatmap.png")
    plt.savefig(heatmap_file)
    plt.show()
    print(f"✅ Heatmap saved: {heatmap_file}")
