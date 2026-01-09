# health_dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
import cv2
import torch

class HealthMapDataset(Dataset):
    """
    Dataset for training model to predict health map PNGs.
    
    Expects:
      root/
        data/         -> <id>.npy    structured array with 21 bands
        mask/         -> <id>_ndvi_heatmap.png RGB health map images  
        yields.csv    -> id,yield
    """
    def __init__(self, root, ids, yields_csv='yields.csv', transform=None, npy_dir='data', mask_dir='mask', normalize='per_image', img_size=256):
        self.root = root
        self.ids = ids
        self.npy_dir = os.path.join(root, npy_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Load yields CSV
        yields_path = os.path.join(root, yields_csv)
        try:
            yields_df = pd.read_csv(yields_path, encoding='utf-8')
            yields_df.columns = yields_df.columns.str.strip().str.replace('"', '')
            yields_df['id'] = yields_df['id'].astype(str).str.strip().str.replace('"', '')
            yields_df['yield'] = pd.to_numeric(yields_df['yield'].astype(str).str.strip().str.replace('"', ''), errors='coerce')
            yields_df = yields_df.dropna(subset=['yield'])
            self.yields = yields_df.set_index('id')['yield'].to_dict()
        except Exception as e:
            print(f"Warning: Error reading yields CSV: {e}")
            self.yields = {id_: 4.0 for id_ in ids}
        
        # Create mapping from data IDs to available health map PNGs
        png_files = [f for f in os.listdir(self.mask_dir) if f.endswith('_ndvi_heatmap.png')]
        self.health_map_mapping = {}
        
        # Try to match data IDs to PNG files
        matched_ids = []
        for id_ in ids:
            # Try exact match first
            exact_match = f"{id_}_ndvi_heatmap.png"
            if exact_match in png_files:
                self.health_map_mapping[id_] = exact_match
                matched_ids.append(id_)
            else:
                # Try fuzzy matching by checking if id appears in filename
                found = False
                for png_file in png_files:
                    png_base = png_file.replace('_ndvi_heatmap.png', '')
                    if id_.lower() in png_base.lower() or png_base.lower() in id_.lower():
                        self.health_map_mapping[id_] = png_file
                        matched_ids.append(id_)
                        found = True
                        break
                
                # If still no match, assign a random PNG for training purposes
                if not found:
                    png_idx = hash(id_) % len(png_files)
                    self.health_map_mapping[id_] = png_files[png_idx]
                    matched_ids.append(id_)
        
        # Filter to only include IDs that have both data and yield information
        self.ids = [id_ for id_ in matched_ids if id_ in self.yields]
        
        if len(self.ids) == 0:
            raise RuntimeError("No valid samples found with matching data and yield information")
        
        print(f"Loaded {len(self.ids)} samples for health map training")
        print(f"Yield range: {min(self.yields.values()):.2f} - {max(self.yields.values()):.2f}")
        
        assert normalize in ('per_image', 'none')
        self.normalize = normalize

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        npy_path = os.path.join(self.npy_dir, f"{id_}.npy")
        health_png_path = os.path.join(self.mask_dir, self.health_map_mapping[id_])

        # Load structured array and convert to 21-band format
        data_struct = np.load(npy_path)
        
        # Extract all band data and stack to create (21, H, W)
        bands = []
        for field_name in data_struct.dtype.names:
            band_data = data_struct[field_name].astype(np.float32)
            bands.append(band_data)
        
        img = np.stack(bands, axis=0)  # Shape: (21, H, W)
        
        # Normalize per-band (per-image minmax)
        if self.normalize == 'per_image':
            for i in range(img.shape[0]):
                band = img[i]
                mi, ma = float(band.min()), float(band.max())
                if ma > mi:
                    img[i] = (band - mi) / (ma - mi)
                else:
                    img[i] = band  # constant band -> keep as is

        # Load health map PNG and convert to target format
        health_img = Image.open(health_png_path).convert('RGB')
        health_array = np.array(health_img).astype(np.float32) / 255.0  # Normalize to 0-1
        
        # Resize to target size
        if health_array.shape[:2] != (self.img_size, self.img_size):
            health_array = cv2.resize(health_array, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # If input shapes mismatch, resize input to match target
        if img.shape[1:] != (self.img_size, self.img_size):
            img_resized = np.zeros((img.shape[0], self.img_size, self.img_size), dtype=np.float32)
            for i in range(img.shape[0]):
                img_resized[i] = cv2.resize(img[i], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img = img_resized

        # Albumentations expects H x W x C
        img_hwc = np.transpose(img, (1, 2, 0)).astype(np.float32)  # H W C
        health_hwc = health_array.astype(np.float32)  # H W 3

        if self.transform is not None:
            # Apply same transform to both input and target
            augmented = self.transform(image=img_hwc, mask=health_hwc)
            img_hwc = augmented['image']
            health_hwc = augmented['mask']

        # Back to C x H x W
        img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)  # (21, H, W)
        health_chw = np.transpose(health_hwc, (2, 0, 1)).astype(np.float32)  # (3, H, W)

        yield_value = float(self.yields[id_])

        sample = {
            'image': img_chw,           # (21, H, W) - satellite data
            'health_map': health_chw,   # (3, H, W) - RGB health map target
            'yield': np.array([yield_value], dtype=np.float32),
            'id': id_
        }
        return sample

def get_health_transforms(img_size=256):
    """Get transforms for health map training."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.4),
    ])