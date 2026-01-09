import os
import json
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt

def read_tif(path):
    try:
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        return img  # (bands, H, W)
    except Exception as e:
        print(f"[⚠️ read_tif] Skipping {path}: {e}")
        return None

def resize_image(img_array, target=(64,64)):
    bands, h, w = img_array.shape
    resized = []
    for b in range(bands):
        resized.append(cv2.resize(img_array[b], target, interpolation=cv2.INTER_LINEAR))
    return np.stack(resized)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def plot_heatmap(labels_grid, save_path):
    plt.figure(figsize=(6,6))
    plt.imshow(labels_grid, cmap="RdYlGn_r")
    plt.colorbar(label="Crop Condition")
    plt.title("Crop Condition Heatmap")
    plt.savefig(save_path)
    plt.close()
