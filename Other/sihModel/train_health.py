# train_health.py
import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from health_dataset import HealthMapDataset, get_health_transforms
from health_model import UNetHealthMapOnly
from health_utils import combined_health_loss
from tqdm import tqdm
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_health_model():
    seed_everything(42)
    
    # Parameters
    root = "data"
    epochs = 20
    batch_size = 4
    img_size = 256
    lr = 1e-4
    save_dir = "health_models"
    
    os.makedirs(save_dir, exist_ok=True)

    # Collect IDs from data folder
    data_dir = os.path.join(root, "data")
    ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".npy")]
    ids = sorted(ids)
    if len(ids) == 0:
        raise RuntimeError("No .npy files found in data/")

    np.random.shuffle(ids)
    split = int(0.8 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    print(f"Training on {len(train_ids)} samples, validating on {len(val_ids)} samples")

    # Create datasets
    train_ds = HealthMapDataset(root, train_ids, yields_csv='yields.csv', 
                               transform=get_health_transforms(img_size), 
                               normalize='per_image', img_size=img_size)
    val_ds = HealthMapDataset(root, val_ids, yields_csv='yields.csv', 
                             transform=get_health_transforms(img_size), 
                             normalize='per_image', img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = UNetHealthMapOnly(in_ch=21, base=32).to(device)
    
    # Print model info
    param_count = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {param_count:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train")
        for batch_idx, batch in enumerate(pbar):
            imgs = batch['image'].to(device, dtype=torch.float32)
            health_maps = batch['health_map'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            
            pred_health = model(imgs)
            loss_t, health_l, yield_l = combined_health_loss(
                pred_health, health_maps, 
                w_health=1.0, w_yield=0.0, health_loss_type='combined'
            )
            
            loss_t.backward()
            optimizer.step()

            running_loss += loss_t.item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        train_loss_epoch = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device, dtype=torch.float32)
                health_maps = batch['health_map'].to(device, dtype=torch.float32)
                
                pred_health = model(imgs)
                loss_v, health_l, yield_l = combined_health_loss(
                    pred_health, health_maps, 
                    w_health=1.0, w_yield=0.0, health_loss_type='combined'
                )
                
                val_loss += loss_v.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss_epoch:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_health_model.pth"))
            print(f"  ✅ Saved best model (val_loss={best_val:.6f})")
    
    print(f"Training completed! Best validation loss: {best_val:.6f}")

if __name__ == "__main__":
    train_health_model()