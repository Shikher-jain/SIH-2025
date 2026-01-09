# main.py
"""
Orchestrator:
1) build dataset from data/
2) train baseline RF and optional CNN-LSTM
3) optionally run one prediction interactively
"""

import os
from modules.dataset_builder import build_dataset
from modules.train_model import train_baseline_rf, train_cnn_lstm
from modules.predict import interactive_predict
from modules.model_cnn_lstm import *
import time

if __name__ == "__main__":
    start = time.time()
    print("=== Agro AI pipeline ===")
    csv_path = build_dataset()  # builds outputs/datasets/cell_features.csv and images_and_seq.npz
    print("Dataset built:", csv_path)   

    print("Training baseline RandomForest...")
    rf = train_baseline_rf(csv_path=csv_path)

    # If you want DL training and you have enough data and GPU, uncomment:
    print("Training CNN+LSTM (this may take long)...")
    cnn = train_cnn_lstm()

    print(time.time() - start)
    
    # interactive prediction (single cell)
    do_pred = input("Run single interactive prediction now? (y/N): ").strip().lower()
    if do_pred == "y":
        interactive_predict()
    else:
        print("Done. Use scheduler.py to auto-train weekly or run predictions manually.")

    