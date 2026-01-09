#!/usr/bin/env python3
"""
Batch converter to create health map PNGs from NPY yield prediction data.
This script processes all mask NPY files and converts them to health map PNGs for training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from pathlib import Path
import argparse

def create_health_colormap():
    """Create a custom colormap for health visualization."""
    colors = ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFFF00', 
              '#ADFF2F', '#32CD32', '#228B22', '#006400', '#004000']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('health', colors, N=n_bins)
    return cmap

def convert_mask_to_health_score(mask_data, method='normalized'):
    """
    Convert mask/yield data to health scores.
    
    Args:
        mask_data: numpy array of mask/yield values
        method: conversion method ('normalized', 'threshold')
    
    Returns:
        numpy array of health scores (0-1)
    """
    # Handle different input types
    if len(mask_data.shape) == 2 and hasattr(mask_data.dtype, 'names') and mask_data.dtype.names is not None:
        # Structured array with RGB
        if 'Red' in mask_data.dtype.names:
            r = mask_data['Red'].astype(np.float32)
            g = mask_data['Green'].astype(np.float32) 
            b = mask_data['Blue'].astype(np.float32)
            gray_data = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            # Use first field
            gray_data = mask_data[mask_data.dtype.names[0]].astype(np.float32)
    else:
        # Regular array
        gray_data = mask_data.astype(np.float32)
    
    # Normalize to health scores
    if method == 'normalized':
        min_val = gray_data.min()
        max_val = gray_data.max()
        if max_val > min_val:
            health_scores = (gray_data - min_val) / (max_val - min_val)
        else:
            health_scores = np.ones_like(gray_data) * 0.5
    elif method == 'threshold':
        # Use thresholds based on data range
        percentiles = np.percentile(gray_data, [20, 40, 60, 80])
        health_scores = np.zeros_like(gray_data)
        health_scores[gray_data >= percentiles[3]] = 1.0
        health_scores[(gray_data >= percentiles[2]) & (gray_data < percentiles[3])] = 0.8
        health_scores[(gray_data >= percentiles[1]) & (gray_data < percentiles[2])] = 0.6
        health_scores[(gray_data >= percentiles[0]) & (gray_data < percentiles[1])] = 0.4
        health_scores[gray_data < percentiles[0]] = 0.2
    else:
        health_scores = np.ones_like(gray_data) * 0.5  # Default fallback
    
    return health_scores

def convert_png_to_health_png(input_png, output_png, method='normalized', size=(256, 256)):
    """
    Convert existing PNG heatmap to health map PNG.
    
    Args:
        input_png: path to input PNG file
        output_png: path to output health PNG file
        method: health conversion method
        size: output image size (H, W)
    """
    from PIL import Image
    import cv2
    
    # Load the PNG image
    img = Image.open(input_png)
    img_array = np.array(img)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            # Use RGB channels and ignore alpha
            rgb = img_array[:, :, :3]
            gray_data = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        elif img_array.shape[2] == 3:  # RGB
            gray_data = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        else:
            gray_data = img_array[:, :, 0]  # Use first channel
    else:
        gray_data = img_array
    
    # Convert to health scores
    health_scores = convert_mask_to_health_score(gray_data, method=method)
    
    # Resize if needed
    if health_scores.shape != size:
        health_scores = cv2.resize(health_scores, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Create health colormap
    health_cmap = create_health_colormap()
    
    # Create figure without axes for clean output
    fig, ax = plt.subplots(1, 1, figsize=(size[1]/100, size[0]/100), dpi=100)
    ax.imshow(health_scores, cmap=health_cmap, vmin=0, vmax=1)
    ax.axis('off')
    
    # Save as PNG with no padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_png, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()

def batch_convert_png_to_healthmaps(data_root, output_dir, method='normalized', size=(256, 256)):
    """
    Batch convert all PNG heatmap files to health map PNGs.
    
    Args:
        data_root: root directory containing mask/ folder with PNG files
        output_dir: directory to save health map PNGs
        method: health conversion method
        size: output image size
    """
    mask_dir = os.path.join(data_root, 'mask')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files
    png_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    print(f"Converting {len(png_files)} PNG files to health maps...")
    print(f"Method: {method}")
    print(f"Output size: {size}")
    print(f"Output directory: {output_dir}")
    
    for png_file in tqdm(png_files, desc="Converting PNGs"):
        png_path = os.path.join(mask_dir, png_file)
        # Create corresponding health PNG filename
        health_file = png_file.replace('_ndvi_heatmap.png', '_health.png')
        output_path = os.path.join(output_dir, health_file)
        
        try:
            convert_png_to_health_png(png_path, output_path, method=method, size=size)
        except Exception as e:
            print(f"Error converting {png_file}: {e}")
            continue
    
    print(f"✅ Conversion complete! Health maps saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert mask NPY files to health map PNGs')
    parser.add_argument('--data_root', default='data', help='Root directory containing mask/ folder')
    parser.add_argument('--output_dir', default='data/health_maps', help='Output directory for health PNGs')
    parser.add_argument('--method', choices=['normalized', 'threshold'], default='normalized', help='Health conversion method')
    parser.add_argument('--size', nargs=2, type=int, default=[256, 256], help='Output image size (H W)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root '{args.data_root}' not found.")
        return 1
    
    mask_dir = os.path.join(args.data_root, 'mask')
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory '{mask_dir}' not found.")
        return 1
    
    batch_convert_png_to_healthmaps(
        args.data_root, 
        args.output_dir, 
        method=args.method,
        size=tuple(args.size)
    )
    
    return 0

if __name__ == '__main__':
    exit(main())