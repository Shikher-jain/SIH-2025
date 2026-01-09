# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def combined_loss(pred_mask, true_mask, pred_yield, true_yield, w_mask=1.0, w_yield=1.0):
    """
    Returns total loss (tensor), mask_loss (float), yield_loss (float)
    mask: MSE, yield: MSE (you can switch to L1)
    """
    mask_loss_t = torch.mean((pred_mask - true_mask) ** 2)
    yield_loss_t = torch.mean((pred_yield - true_yield) ** 2)
    total = w_mask * mask_loss_t + w_yield * yield_loss_t
    return total, float(mask_loss_t.item()), float(yield_loss_t.item())

def save_heatmap_advanced(pred_mask, out_path, predicted_yield=None, true_yield=None, sample_id=None):
    """
    Advanced heatmap visualization with multiple views and enhanced features
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pred = np.clip(pred_mask, 0.0, 1.0)
    
    # Calculate statistics
    stats = {
        'min': pred.min(),
        'max': pred.max(), 
        'mean': pred.mean(),
        'std': pred.std(),
        'range': pred.max() - pred.min()
    }
    
    print(f"Heatmap stats - Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.suptitle(f'Crop Yield Prediction Analysis{f" - {sample_id}" if sample_id else ""}', fontsize=16, fontweight='bold')
    
    # 1. Adaptive range heatmap
    ax1 = plt.subplot(2, 3, 1)
    vmin_adaptive = stats['min']
    vmax_adaptive = stats['max']
    if stats['range'] < 0.1:
        center = stats['mean']
        vmin_adaptive = max(0, center - 0.05)
        vmax_adaptive = min(1, center + 0.05)
    
    im1 = ax1.imshow(pred, cmap='hot', vmin=vmin_adaptive, vmax=vmax_adaptive)
    ax1.set_title(f'Adaptive Range\n[{vmin_adaptive:.3f}, {vmax_adaptive:.3f}]')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Full range heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(pred, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('Full Range [0, 1]')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. Alternative colormap (viridis for comparison)
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(pred, cmap='viridis', vmin=vmin_adaptive, vmax=vmax_adaptive)
    ax3.set_title('Viridis Colormap\n(Adaptive Range)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 4. Histogram of prediction values
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(pred.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.3f}')
    ax4.axvline(stats['min'], color='blue', linestyle='--', label=f'Min: {stats["min"]:.3f}')
    ax4.axvline(stats['max'], color='green', linestyle='--', label=f'Max: {stats["max"]:.3f}')
    ax4.set_xlabel('Prediction Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Predictions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Contour plot
    ax5 = plt.subplot(2, 3, 5)
    contour = ax5.contour(pred, levels=10, colors='black', alpha=0.6, linewidths=0.5)
    contourf = ax5.contourf(pred, levels=20, cmap='hot', vmin=vmin_adaptive, vmax=vmax_adaptive)
    ax5.clabel(contour, inline=True, fontsize=8)
    ax5.set_title('Contour Plot')
    ax5.axis('off')
    plt.colorbar(contourf, ax=ax5, shrink=0.8)
    
    # 6. Statistics and yield comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create text summary
    summary_text = f"""
PREDICTION STATISTICS:
• Min Value: {stats['min']:.4f}
• Max Value: {stats['max']:.4f}
• Mean Value: {stats['mean']:.4f} 
• Std Deviation: {stats['std']:.4f}
• Value Range: {stats['range']:.4f}
• Non-zero pixels: {np.count_nonzero(pred > 0.1):,} / {pred.size:,}
"""
    
    if predicted_yield is not None:
        summary_text += f"\nYIELD PREDICTION:\n• Predicted Yield: {predicted_yield:.3f}"
        
    if true_yield is not None:
        summary_text += f"\n• True Yield: {true_yield:.3f}"
        if predicted_yield is not None:
            error = abs(predicted_yield - true_yield)
            error_pct = (error / true_yield) * 100
            summary_text += f"\n• Absolute Error: {error:.3f}\n• Error %: {error_pct:.1f}%"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Also create a clean single-panel version
    clean_path = out_path.replace('.png', '_clean.png')
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(pred, cmap='hot', vmin=vmin_adaptive, vmax=vmax_adaptive)
    cbar = plt.colorbar(shrink=0.8, label='Crop Yield Prediction')
    cbar.ax.tick_params(labelsize=12)
    
    title = 'Crop Yield Prediction Heatmap'
    if sample_id:
        title += f'\nSample: {sample_id}'
    if predicted_yield:
        title += f'\nPredicted Yield: {predicted_yield:.3f}'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(clean_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved advanced heatmap -> {out_path}")
    print(f"Saved clean heatmap -> {clean_path}")
    
    return stats

def compute_yield_metrics(y_true_list, y_pred_list):
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae}

def save_heatmap(pred_mask, out_path, adaptive_range=True):
    """
    Simple heatmap function for backward compatibility
    """
    return save_heatmap_advanced(pred_mask, out_path)

def save_focused_heatmap(pred_mask, out_path, predicted_yield=None, sample_id=None):
    """
    Create a clean, focused heatmap visualization
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pred = np.clip(pred_mask, 0.0, 1.0)
    
    # Calculate adaptive range for better contrast
    vmin = pred.min()
    vmax = pred.max()
    
    # Create the main heatmap
    plt.figure(figsize=(12, 10), dpi=150)
    
    # Use a clean, professional colormap
    im = plt.imshow(pred, cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    # Add colorbar with custom styling
    cbar = plt.colorbar(im, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Crop Yield Prediction', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Create title
    title_parts = ['Crop Yield Prediction Heatmap']
    if sample_id:
        title_parts.append(f'Sample: {sample_id}')
    if predicted_yield:
        title_parts.append(f'Predicted Yield: {predicted_yield:.3f}')
    
    plt.title('\n'.join(title_parts), fontsize=16, fontweight='bold', pad=20)
    
    # Remove axes for cleaner look
    plt.axis('off')
    
    # Add subtle grid overlay for better spatial reference
    ax = plt.gca()
    ax.set_xticks(np.arange(0, pred.shape[1], 32))
    ax.set_yticks(np.arange(0, pred.shape[0], 32))
    ax.grid(True, alpha=0.3, linewidth=0.5, color='white')
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    print(f"Heatmap visualization saved -> {out_path}")
    print(f"Value range: [{vmin:.4f}, {vmax:.4f}], Mean: {pred.mean():.4f}")
    
    return {'min': vmin, 'max': vmax, 'mean': pred.mean(), 'std': pred.std()}
