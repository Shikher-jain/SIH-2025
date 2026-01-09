import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.utils import Sequence

class YieldHeatmapDataset(Sequence):
    def __init__(self, image_dir, mask_dir, csv_path, batch_size=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.data = pd.read_csv(csv_path)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch = self.data.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]
        images, masks, yields = [], [], []

        for _, row in batch.iterrows():
            fname = row['filename']
            image = np.load(os.path.join(self.image_dir, fname + '.npy'))
            mask = cv2.imread(os.path.join(self.mask_dir, fname + '_mask.png'), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512)) / 255.0
            mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask)
            yields.append(row['yield'])

        return np.array(images), {"heatmap_output": np.array(masks), "yield_output": np.array(yields)}
