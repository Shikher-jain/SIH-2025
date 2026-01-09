# health_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def health_map_loss(pred_health, target_health, loss_type='mse'):
    """
    Loss function for RGB health map prediction.
    
    Args:
        pred_health: predicted health map (B, 3, H, W)
        target_health: target health map (B, 3, H, W)
        loss_type: 'mse', 'l1', 'combined'
    
    Returns:
        loss tensor
    """
    if loss_type == 'mse':
        return F.mse_loss(pred_health, target_health)
    elif loss_type == 'l1':
        return F.l1_loss(pred_health, target_health)
    elif loss_type == 'combined':
        mse_loss = F.mse_loss(pred_health, target_health)
        l1_loss = F.l1_loss(pred_health, target_health)
        return 0.7 * mse_loss + 0.3 * l1_loss
    else:
        return F.mse_loss(pred_health, target_health)

def combined_health_loss(pred_health, target_health, pred_yield=None, target_yield=None, 
                        w_health=1.0, w_yield=0.1, health_loss_type='combined'):
    """
    Combined loss for health map and optional yield prediction.
    
    Args:
        pred_health: predicted health map (B, 3, H, W)
        target_health: target health map (B, 3, H, W)
        pred_yield: predicted yield (B, 1) - optional
        target_yield: target yield (B, 1) - optional
        w_health: weight for health map loss
        w_yield: weight for yield loss
        health_loss_type: type of health map loss
    
    Returns:
        total_loss, health_loss, yield_loss
    """
    health_loss = health_map_loss(pred_health, target_health, health_loss_type)
    
    if pred_yield is not None and target_yield is not None:
        yield_loss = F.mse_loss(pred_yield, target_yield)
        total_loss = w_health * health_loss + w_yield * yield_loss
        return total_loss, health_loss, yield_loss
    else:
        return health_loss, health_loss, torch.tensor(0.0)

def save_health_map_png(health_tensor, output_path, denormalize=True):
    """
    Save a health map tensor as PNG image.
    
    Args:
        health_tensor: tensor of shape (3, H, W) or (H, W, 3)
        output_path: path to save PNG
        denormalize: whether to denormalize from [0,1] to [0,255]
    """
    if isinstance(health_tensor, torch.Tensor):
        health_array = health_tensor.detach().cpu().numpy()
    else:
        health_array = health_tensor
    
    # Ensure correct shape (H, W, 3)
    if health_array.shape[0] == 3:  # (3, H, W)
        health_array = np.transpose(health_array, (1, 2, 0))
    
    # Denormalize if needed
    if denormalize:
        health_array = (health_array * 255).astype(np.uint8)
    else:
        health_array = health_array.astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(health_array, 'RGB')
    img.save(output_path)

def create_health_comparison_plot(input_img, pred_health, target_health, save_path=None):
    """
    Create a comparison plot showing input, prediction, and target.
    
    Args:
        input_img: input satellite image (21, H, W) - will show RGB channels
        pred_health: predicted health map (3, H, W)
        target_health: target health map (3, H, W)
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy if needed
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    if isinstance(pred_health, torch.Tensor):
        pred_health = pred_health.detach().cpu().numpy()
    if isinstance(target_health, torch.Tensor):
        target_health = target_health.detach().cpu().numpy()
    
    # Show input (use first 3 channels as RGB approximation)
    if input_img.shape[0] >= 3:
        rgb_approx = np.transpose(input_img[:3], (1, 2, 0))
        rgb_approx = (rgb_approx - rgb_approx.min()) / (rgb_approx.max() - rgb_approx.min())
        axes[0].imshow(rgb_approx)
    else:
        axes[0].imshow(input_img[0], cmap='gray')
    axes[0].set_title('Input (RGB Approx)')
    axes[0].axis('off')
    
    # Show prediction
    pred_rgb = np.transpose(pred_health, (1, 2, 0))
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Predicted Health Map')
    axes[1].axis('off')
    
    # Show target
    target_rgb = np.transpose(target_health, (1, 2, 0))
    axes[2].imshow(target_rgb)
    axes[2].set_title('Target Health Map')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def health_map_metrics(pred_health, target_health):
    """
    Calculate metrics for health map prediction.
    
    Args:
        pred_health: predicted health map (B, 3, H, W)
        target_health: target health map (B, 3, H, W)
    
    Returns:
        dict of metrics
    """
    with torch.no_grad():
        mse = F.mse_loss(pred_health, target_health)
        mae = F.l1_loss(pred_health, target_health)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # SSIM approximation (per channel)
        pred_flat = pred_health.view(pred_health.size(0), pred_health.size(1), -1)
        target_flat = target_health.view(target_health.size(0), target_health.size(1), -1)
        
        # Correlation coefficient per channel
        correlation = []
        for ch in range(pred_health.size(1)):
            pred_ch = pred_flat[:, ch, :]
            target_ch = target_flat[:, ch, :]
            corr = torch.corrcoef(torch.stack([pred_ch.flatten(), target_ch.flatten()]))[0, 1]
            correlation.append(corr.item())
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'psnr': psnr.item(),
            'correlation_rgb': correlation,
            'correlation_mean': np.mean(correlation)
        }