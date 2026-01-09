# dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
import cv2

class MultiNPYDataset(Dataset):
    """
    Expects:
      root/
        data/         -> <id>.npy    structured array with 21 bands
        mask/         -> satellite_data_N.npy RGB structured arrays  
        yields.csv    -> id,yield
    """
    def __init__(self, root, ids, yields_csv='yields.csv', transform=None, npy_dir='data', mask_dir='mask', normalize='per_image'):
        self.root = root
        self.ids = ids
        self.npy_dir = os.path.join(root, npy_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.transform = transform
        # Load yields CSV with robust parsing
        yields_path = os.path.join(root, yields_csv)
        try:
            # Try loading with error handling
            yields_df = pd.read_csv(yields_path, encoding='utf-8')
            # Clean column names (remove quotes and spaces)
            yields_df.columns = yields_df.columns.str.strip().str.replace('"', '')
            # Clean the data
            yields_df['id'] = yields_df['id'].astype(str).str.strip().str.replace('"', '')
            yields_df['yield'] = pd.to_numeric(yields_df['yield'].astype(str).str.strip().str.replace('"', ''), errors='coerce')
            # Remove rows with NaN yields
            yields_df = yields_df.dropna(subset=['yield'])
            self.yields = yields_df.set_index('id')['yield'].to_dict()
        except Exception as e:
            print(f"Warning: Error reading yields CSV: {e}")
            print("Using default yield values for all samples")
            # Fallback: assign default yield values
            self.yields = {id_: 4.0 for id_ in ids}
        
        # Filter IDs to only include those with yield data
        self.ids = [id_ for id_ in ids if id_ in self.yields]
        if len(self.ids) == 0:
            raise RuntimeError("No valid samples found with matching yield data")
        
        print(f"Loaded {len(self.ids)} samples with yield data")
        print(f"Yield range: {min(self.yields.values()):.2f} - {max(self.yields.values()):.2f}")
        assert normalize in ('per_image', 'none')
        self.normalize = normalize
        
        # Get available mask files and create mapping
        mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith('.npy')]
        self.mask_mapping = {}
        for i, id_ in enumerate(self.ids):
            # Try to find exact match first
            exact_match = f'{id_}_mask.npy'
            if exact_match in mask_files:
                self.mask_mapping[id_] = exact_match
            else:
                # Fallback to cycling through available masks
                mask_idx = (i % len(mask_files))
                self.mask_mapping[id_] = mask_files[mask_idx]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        npy_path = os.path.join(self.npy_dir, f"{id_}.npy")
        mask_path = os.path.join(self.mask_dir, self.mask_mapping[id_])

        # Load structured array and convert to 21-band format
        data_struct = np.load(npy_path)
        
        # Extract all band data and stack to create (21, H, W)
        bands = []
        for field_name in data_struct.dtype.names:
            band_data = data_struct[field_name].astype(np.float32)
            bands.append(band_data)
        
        img = np.stack(bands, axis=0)  # Shape: (21, H, W)
        
        # normalize per-band (per-image minmax)
        if self.normalize == 'per_image':
            for i in range(img.shape[0]):
                band = img[i]
                mi, ma = float(band.min()), float(band.max())
                if ma > mi:
                    img[i] = (band - mi) / (ma - mi)
                else:
                    img[i] = band  # constant band -> keep as is

        # Load mask structured array and convert to grayscale
        mask_struct = np.load(mask_path)
        # Convert RGB to grayscale using standard weights
        r = mask_struct['Red'].astype(np.float32)
        g = mask_struct['Green'].astype(np.float32) 
        b = mask_struct['Blue'].astype(np.float32)
        mask_gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Normalize to 0-1 range
        mask = mask_gray / mask_gray.max() if mask_gray.max() > 0 else mask_gray
        mask = np.expand_dims(mask, axis=0)  # 1 x H x W

        # If shapes mismatch, resize input to match mask
        if img.shape[1:] != mask.shape[1:]:
            _, h_t, w_t = mask.shape
            img_resized = np.zeros((img.shape[0], h_t, w_t), dtype=np.float32)
            for i in range(img.shape[0]):
                img_resized[i] = cv2.resize(img[i], (w_t, h_t), interpolation=cv2.INTER_LINEAR)
            img = img_resized

        # Albumentations expects H x W x C
        img_hwc = np.transpose(img, (1, 2, 0)).astype(np.float32)  # H W C
        mask_hwc = np.transpose(mask, (1, 2, 0)).astype(np.float32)  # H W 1

        if self.transform is not None:
            augmented = self.transform(image=img_hwc, mask=mask_hwc)
            img_hwc = augmented['image']
            mask_hwc = augmented['mask']

        # back to C x H x W
        img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
        mask_chw = np.transpose(mask_hwc, (2, 0, 1)).astype(np.float32)

        yield_value = float(self.yields[id_])

        sample = {
            'image': img_chw,           # (21, H, W)
            'mask': mask_chw,           # (1, H, W)
            'yield': np.array([yield_value], dtype=np.float32),
            'id': id_
        }
        return sample

def get_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.4),
    ])
