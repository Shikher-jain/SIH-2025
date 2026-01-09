# train.py
import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import MultiNPYDataset, get_transforms
from model import UNetMultiHead
from utils import combined_loss
from tqdm import tqdm
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_main(root="data",
               epochs=30,
               batch_size=8,
               img_size=256,
               lr=1e-4,
               save_dir="models"):
    seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)

    # collect ids from data folder
    data_dir = os.path.join(root, "data")
    ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".npy")]
    ids = sorted(ids)
    if len(ids) == 0:
        raise RuntimeError("No .npy files found in data/")

    np.random.shuffle(ids)
    split = int(0.8 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_ds = MultiNPYDataset(root, train_ids, yields_csv='yields.csv', transform=get_transforms(img_size), normalize='per_image')
    val_ds = MultiNPYDataset(root, val_ids, yields_csv='yields.csv', transform=get_transforms(img_size), normalize='per_image')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetMultiHead(in_ch=21, base=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for batch in pbar:
            imgs = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].to(device, dtype=torch.float32)
            yields = batch['yield'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            pred_mask, pred_yield = model(imgs)
            loss_t, mask_l, yield_l = combined_loss(pred_mask, masks, pred_yield, yields, w_mask=1.0, w_yield=1.0)
            loss_t.backward()
            optimizer.step()

            running_loss += loss_t.item()
            pbar.set_postfix(loss=(running_loss / (pbar.n + 1)), mask=float(mask_l), yield_loss=float(yield_l))

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device, dtype=torch.float32)
                masks = batch['mask'].to(device, dtype=torch.float32)
                yields = batch['yield'].to(device, dtype=torch.float32)
                pred_mask, pred_yield = model(imgs)
                loss_v, _, _ = combined_loss(pred_mask, masks, pred_yield, yields, w_mask=1.0, w_yield=1.0)
                val_loss += loss_v.item()
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch} validation loss: {val_loss:.6f}")
        scheduler.step(val_loss)

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pth"))
            print(f"Saved best model (val_loss={best_val:.6f})")

if __name__ == "__main__":
    train_main(root="data", epochs=30, batch_size=8, img_size=256)
