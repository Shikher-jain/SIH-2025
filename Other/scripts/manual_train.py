#!/usr/bin/env python3
"""
Standalone Manual Training Script with Interactive Parameters
Direct model training without API dependencies
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_training_parameters():
    """Interactive parameter input with defaults"""
    print("🔧 Training Configuration")
    print("=" * 30)
    
    try:
        epochs = int(input("Enter number of epochs (default: 50): ") or "50")
    except ValueError:
        epochs = 50
        
    try:
        batch_size = int(input("Enter batch size (default: 4): ") or "4")
    except ValueError:
        batch_size = 4
        
    try:
        learning_rate = float(input("Enter learning rate (default: 0.001): ") or "0.001")
    except ValueError:
        learning_rate = 0.001
    
    print(f"\n✅ Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    
    return epochs, batch_size, learning_rate

# Simple configuration
INPUT_SHAPE = (512, 512, 21)

class AgriDataset(Dataset):
    """PyTorch Dataset for agricultural data"""
    def __init__(self, X, Y_mask, Y_yield):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # Convert to CHW format
        self.Y_mask = torch.FloatTensor(Y_mask).permute(0, 3, 1, 2) if Y_mask.ndim == 4 else torch.FloatTensor(Y_mask).unsqueeze(1)
        self.Y_yield = torch.FloatTensor(Y_yield)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y_mask[idx], self.Y_yield[idx]

class UNetModel(nn.Module):
    """PyTorch U-Net model for segmentation and yield prediction"""
    def __init__(self, input_channels=21):
        super(UNetModel, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.dec3 = self._conv_block(256 + 128, 128)
        self.dec2 = self._conv_block(128 + 64, 64)
        self.dec1 = self._conv_block(64 + 32, 32)
        
        # Output layers
        self.mask_output = nn.Conv2d(32, 1, kernel_size=1)
        
        # Yield prediction branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.yield_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Yield prediction from bottleneck
        yield_features = self.global_pool(b).view(b.size(0), -1)
        yield_pred = self.yield_fc(yield_features)
        
        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Mask output
        mask_pred = torch.sigmoid(self.mask_output(d1))
        
        return mask_pred, yield_pred

def load_yield_data(csv_path):
    """Load yield data from CSV"""
    if not os.path.exists(csv_path):
        print(f"❌ Yield CSV not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    df['ID'] = df['ID'].str.strip()
    df['yield'] = pd.to_numeric(df['yield'], errors='coerce')
    
    # Extract district names
    df['district'] = df['ID'].str.replace(r'\d+', '', regex=True).str.strip().str.lower()
    yield_dict = df.groupby('district')['yield'].first().to_dict()
    
    print(f"✅ Loaded {len(yield_dict)} districts from CSV")
    return yield_dict

def load_training_data(input_folder, mask_folder, yield_dict):
    """Load training data from folders"""
    X, Y_mask, Y_yield = [], [], []
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        return None, None, None
    
    if not os.path.exists(mask_folder):
        print(f"❌ Mask folder not found: {mask_folder}")
        return None, None, None
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    print(f"📁 Found {len(input_files)} input files")
    
    for file in input_files:
        district_name = file.replace('.npy', '').lower()
        district_name = ''.join([c for c in district_name if not c.isdigit()]).strip()
        
        input_path = os.path.join(input_folder, file)
        mask_path = os.path.join(mask_folder, file)
        
        if not os.path.exists(mask_path):
            continue
        
        # Get yield value
        if district_name not in yield_dict:
            continue
        
        try:
            # Load input data (per-band min-max normalization)
            x = np.load(input_path).astype('float32')
            # Per-band min-max normalization as specified in memory
            for band in range(x.shape[-1]):
                band_data = x[:, :, band]
                x[:, :, band] = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)
            
            if x.shape != INPUT_SHAPE:
                continue
            
            # Load mask data
            mask = np.load(mask_path).astype('float32')
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)
            
            if mask.shape[:2] != INPUT_SHAPE[:2]:
                continue
            
            X.append(x)
            Y_mask.append(mask)
            Y_yield.append(yield_dict[district_name])
            
        except Exception as e:
            print(f"⚠️ Error loading {file}: {str(e)}")
            continue
    
    if len(X) == 0:
        print("❌ No valid training data found!")
        return None, None, None
    
    X = np.array(X)
    Y_mask = np.array(Y_mask)
    Y_yield = np.array(Y_yield).reshape(-1, 1)
    
    print(f"✅ Loaded {len(X)} training samples")
    print(f"📐 Input shape: {X.shape}")
    print(f"🎯 Mask shape: {Y_mask.shape}")
    print(f"📊 Yield shape: {Y_yield.shape}")
    
    return X, Y_mask, Y_yield

def train_model():
    """Main training function with interactive parameters"""
    print("🌾 Standalone Manual Model Training")
    print("=" * 40)
    
    # Get interactive parameters
    EPOCHS, BATCH_SIZE, LEARNING_RATE = get_training_parameters()
    
    # Paths - relative to the scripts directory
    input_folder = "../model/data/data"
    mask_folder = "../model/data/mask"
    csv_path = "../model/data/yield.csv"
    model_path = "../model/models/trained_model.pth"
    
    # Create models directory
    os.makedirs("../model/models", exist_ok=True)
    
    # Verify data directories
    print(f"\n🔍 Checking data availability...")
    if not os.path.exists(input_folder):
        print(f"❌ NPY data directory not found: {input_folder}")
        return
    if not os.path.exists(mask_folder):
        print(f"❌ Mask directory not found: {mask_folder}")
        return
    if not os.path.exists(csv_path):
        print(f"❌ Yield CSV not found: {csv_path}")
        return
    
    print("✅ All required directories and files found")
    
    # Load data
    print("📊 Loading yield data...")
    yield_dict = load_yield_data(csv_path)
    if yield_dict is None:
        return
    
    print("📂 Loading training data...")
    X, Y_mask, Y_yield = load_training_data(input_folder, mask_folder, yield_dict)
    
    if X is None:
        print("❌ Failed to load training data")
        return
    
    # Build model
    print("🧠 Building model...")
    model = UNetModel(input_channels=21)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🖥️ Using device: {device}")
    
    # Setup optimizer and loss functions with stability improvements
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    mask_criterion = nn.BCELoss()
    yield_criterion = nn.MSELoss()
    
    print(f"📋 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Split data
    X_train, X_val, Y_mask_train, Y_mask_val, Y_yield_train, Y_yield_val = train_test_split(
        X, Y_mask, Y_yield, test_size=0.2, random_state=42
    )
    
    print(f"🚀 Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    
    # Create datasets and dataloaders
    train_dataset = AgriDataset(X_train, Y_mask_train, Y_yield_train)
    val_dataset = AgriDataset(X_val, Y_mask_val, Y_yield_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mask_acc': [], 'val_mask_acc': [],
        'train_yield_mae': [], 'val_yield_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n🚀 Starting training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mask_acc = 0.0
        train_yield_mae = 0.0
        
        for batch_idx, (data, mask_target, yield_target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            data = data.to(device)
            mask_target = mask_target.to(device)
            yield_target = yield_target.to(device)
            
            optimizer.zero_grad()
            
            mask_pred, yield_pred = model(data)
            
            mask_loss = mask_criterion(mask_pred, mask_target)
            yield_loss = yield_criterion(yield_pred, yield_target)
            total_loss = mask_loss + 0.1 * yield_loss  # Same weights as TF version
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                mask_acc = ((mask_pred > 0.5) == (mask_target > 0.5)).float().mean()
                yield_mae = torch.abs(yield_pred - yield_target).mean()
                train_mask_acc += mask_acc.item()
                train_yield_mae += yield_mae.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mask_acc = 0.0
        val_yield_mae = 0.0
        
        with torch.no_grad():
            for data, mask_target, yield_target in val_loader:
                data = data.to(device)
                mask_target = mask_target.to(device)
                yield_target = yield_target.to(device)
                
                mask_pred, yield_pred = model(data)
                
                mask_loss = mask_criterion(mask_pred, mask_target)
                yield_loss = yield_criterion(yield_pred, yield_target)
                total_loss = mask_loss + 0.1 * yield_loss
                
                val_loss += total_loss.item()
                
                mask_acc = ((mask_pred > 0.5) == (mask_target > 0.5)).float().mean()
                yield_mae = torch.abs(yield_pred - yield_target).mean()
                val_mask_acc += mask_acc.item()
                val_yield_mae += yield_mae.item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mask_acc /= len(train_loader)
        val_mask_acc /= len(val_loader)
        train_yield_mae /= len(train_loader)
        val_yield_mae /= len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mask_acc'].append(train_mask_acc)
        history['val_mask_acc'].append(val_mask_acc)
        history['train_yield_mae'].append(train_yield_mae)
        history['val_yield_mae'].append(val_yield_mae)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Mask Acc: {train_mask_acc:.4f}, Val Mask Acc: {val_mask_acc:.4f}")
        print(f"  Train Yield MAE: {train_yield_mae:.4f}, Val Yield MAE: {val_yield_mae:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': {
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE
                }
            }, model_path)
            print(f"💾 Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"💾 Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_yield_mae'], label='Training MAE')
    plt.plot(history['val_yield_mae'], label='Validation MAE')
    plt.title('Yield MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_mask_acc'], label='Training Accuracy')
    plt.plot(history['val_mask_acc'], label='Validation Accuracy')
    plt.title('Mask Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../model/training_history.png', dpi=150, bbox_inches='tight')
    print("📈 Training plots saved to ../model/training_history.png")
    
    # Final results
    final_loss = history['val_loss'][-1] if history['val_loss'] else 0
    final_mae = history['val_yield_mae'][-1] if history['val_yield_mae'] else 0
    final_acc = history['val_mask_acc'][-1] if history['val_mask_acc'] else 0
    
    print("\n✅ Training completed!")
    print(f"📊 Final Validation Loss: {final_loss:.4f}")
    print(f"📊 Final Yield MAE: {final_mae:.4f}")
    print(f"📊 Final Mask Accuracy: {final_acc:.4f}")
    print(f"📁 Model ready at: {model_path}")

if __name__ == "__main__":
    train_model()