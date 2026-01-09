"""
Generate spatial crop health maps from .npy satellite imagery data.
This script creates visual health maps with color-coded regions showing crop health status.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image
import os
import argparse
from datetime import datetime
import pandas as pd

class CropHealthMapGenerator:
    def __init__(self):
        self.health_colors = {
            'Excellent': '#2E8B57',  # Sea Green
            'Good': '#32CD32',       # Lime Green
            'Fair': '#FFD700',       # Gold
            'Poor': '#FF8C00',       # Dark Orange
            'Critical': '#DC143C'    # Crimson
        }
        
        self.health_thresholds = {
            'Excellent': (0.8, 1.0),
            'Good': (0.65, 0.8),
            'Fair': (0.5, 0.65),
            'Poor': (0.35, 0.5),
            'Critical': (0.0, 0.35)
        }
    
    def load_image_data(self, npy_path):
        """Load and preprocess .npy image data."""
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Image file not found: {npy_path}")
        
        # Load the multispectral image data
        img_data = np.load(npy_path)
        print(f"Loaded image shape: {img_data.shape}")
        
        return img_data
    
    def calculate_vegetation_indices(self, img_data):
        """Calculate vegetation indices from multispectral data."""
        if len(img_data.shape) == 4:
            img_data = img_data[0]  # Remove batch dimension
        if img_data.shape[-1] == 1:
            img_data = img_data[:, :, :, 0]  # Remove channel dimension for hyperspectral
        
        if len(img_data.shape) == 3 and img_data.shape[2] >= 4:
            # Standard multispectral case
            blue = img_data[:, :, 0].astype(np.float32)
            green = img_data[:, :, 1].astype(np.float32)
            red = img_data[:, :, 2].astype(np.float32)
            nir = img_data[:, :, 3].astype(np.float32)
        elif len(img_data.shape) == 3 and img_data.shape[2] >= 30:
            # Hyperspectral case - select specific bands
            blue = img_data[:, :, 10].astype(np.float32)   # Blue around band 10
            green = img_data[:, :, 20].astype(np.float32)  # Green around band 20
            red = img_data[:, :, 30].astype(np.float32)    # Red around band 30
            nir = img_data[:, :, 40].astype(np.float32)    # NIR around band 40
        else:
            raise ValueError(f"Unsupported image shape: {img_data.shape}. Need at least 4 bands.")
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        # Calculate NDVI (Normalized Difference Vegetation Index)
        ndvi = (nir - red) / (nir + red + epsilon)
        ndvi = np.clip(ndvi, -1, 1)
        
        # Calculate EVI (Enhanced Vegetation Index) - simplified version
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)
        evi = np.clip(evi, -1, 1)
        
        # Calculate SAVI (Soil Adjusted Vegetation Index)
        L = 0.5  # Soil brightness correction factor
        savi = ((nir - red) / (nir + red + L)) * (1 + L)
        savi = np.clip(savi, -1, 1)
        
        # Calculate NDWI (Normalized Difference Water Index)
        ndwi = (green - nir) / (green + nir + epsilon)
        ndwi = np.clip(ndwi, -1, 1)
        
        # Calculate GNDVI (Green Normalized Difference Vegetation Index)
        gndvi = (nir - green) / (nir + green + epsilon)
        gndvi = np.clip(gndvi, -1, 1)
        
        return {
            'ndvi': ndvi,
            'evi': evi,
            'savi': savi,
            'ndwi': ndwi,
            'gndvi': gndvi
        }
    
    def calculate_health_score(self, indices, weights=None):
        """Calculate overall health score from vegetation indices using memory specification."""
        if weights is None:
            # Using the health score formula from memory: 0.5 × NDVI + 0.3 × EVI + 0.2 × SAVI
            weights = {'ndvi': 0.5, 'evi': 0.3, 'savi': 0.2}
        
        # Normalize indices to 0-1 range as specified in memory: (Index + 1) / 2
        ndvi_norm = (indices['ndvi'] + 1) / 2  # NDVI: -1 to 1 -> 0 to 1
        evi_norm = (indices['evi'] + 1) / 2    # EVI: -1 to 1 -> 0 to 1
        savi_norm = (indices['savi'] + 1) / 2  # SAVI: -1 to 1 -> 0 to 1
        
        # Calculate weighted health score according to memory specification
        health_score = (weights['ndvi'] * ndvi_norm + 
                       weights['evi'] * evi_norm + 
                       weights['savi'] * savi_norm)
        
        return np.clip(health_score, 0, 1)
    
    def classify_health(self, health_score):
        """Classify pixels into health categories."""
        classification = np.full(health_score.shape, 'Critical', dtype=object)
        
        for health_class, (min_val, max_val) in self.health_thresholds.items():
            mask = (health_score >= min_val) & (health_score < max_val)
            classification[mask] = health_class
        
        # Handle the upper boundary for Excellent
        excellent_mask = health_score >= 0.8
        classification[excellent_mask] = 'Excellent'
        
        return classification
    
    def create_health_map(self, npy_path, output_path=None, title=None, show_stats=True):
        """Create and save a comprehensive crop health map."""
        # Load image data
        img_data = self.load_image_data(npy_path)
        
        # Calculate vegetation indices
        indices = self.calculate_vegetation_indices(img_data)
        
        # Calculate health score
        health_score = self.calculate_health_score(indices)
        
        # Classify health
        health_classification = self.classify_health(health_score)
        
        # Create the visualization with enhanced layout
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # Original RGB image (if available)
        if len(img_data.shape) == 3 and img_data.shape[2] >= 3:
            if img_data.shape[2] >= 30:  # Hyperspectral case
                rgb_img = img_data[:, :, [30, 20, 10]]  # R, G, B from hyperspectral
            else:  # Standard multispectral
                rgb_img = img_data[:, :, [2, 1, 0]]  # Red, Green, Blue
            rgb_img = np.clip(rgb_img, 0, 1)
            axes[0, 0].imshow(rgb_img)
            axes[0, 0].set_title('RGB Composite', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
        
        # NDVI
        im1 = axes[0, 1].imshow(indices['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 1].set_title('NDVI', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        
        # EVI
        im2 = axes[0, 2].imshow(indices['evi'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 2].set_title('EVI', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
        
        # SAVI
        im3 = axes[1, 0].imshow(indices['savi'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[1, 0].set_title('SAVI', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        
        # NDWI
        if 'ndwi' in indices:
            im4 = axes[1, 1].imshow(indices['ndwi'], cmap='Blues', vmin=-1, vmax=1)
            axes[1, 1].set_title('NDWI (Water Index)', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        
        # GNDVI
        if 'gndvi' in indices:
            im5 = axes[1, 2].imshow(indices['gndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
            axes[1, 2].set_title('GNDVI', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)
        
        # Health Score
        im6 = axes[2, 0].imshow(health_score, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2, 0].set_title('Health Score', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        plt.colorbar(im6, ax=axes[2, 0], shrink=0.8)
        
        # Health Classification Map
        class_colors = [self.health_colors[cls] for cls in ['Critical', 'Poor', 'Fair', 'Good', 'Excellent']]
        cmap = colors.ListedColormap(class_colors)
        
        # Convert classification to numeric values
        class_numeric = np.zeros_like(health_score)
        for i, health_class in enumerate(['Critical', 'Poor', 'Fair', 'Good', 'Excellent']):
            mask = health_classification == health_class
            class_numeric[mask] = i
        
        im7 = axes[2, 1].imshow(class_numeric, cmap=cmap, vmin=0, vmax=4)
        axes[2, 1].set_title('Health Classification', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
        
        # Add legend for health classification
        legend_elements = []
        for i, health_class in enumerate(['Critical', 'Poor', 'Fair', 'Good', 'Excellent']):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=class_colors[i], label=health_class))
        axes[2, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Combined health visualization
        combined_health = self.create_combined_health_visualization(indices)
        im8 = axes[2, 2].imshow(combined_health, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2, 2].set_title('Combined Health Index', fontsize=12, fontweight='bold')
        axes[2, 2].axis('off')
        plt.colorbar(im8, ax=axes[2, 2], shrink=0.8)
        
        # Set main title
        if title is None:
            title = f"Comprehensive Crop Health Analysis - {os.path.basename(npy_path)}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Add statistics if requested
        if show_stats:
            stats_text = self.calculate_statistics(health_score, health_classification)
            fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the map
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            output_path = f"../outputs/comprehensive_healthmap_{base_name}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive health map saved to: {output_path}")
        
        plt.show()
        
        return {
            'health_score': health_score,
            'classification': health_classification,
            'indices': indices,
            'output_path': output_path
        }
    
    def calculate_statistics(self, health_score, health_classification):
        """Calculate and format health statistics."""
        # Health score statistics
        mean_health = np.mean(health_score)
        std_health = np.std(health_score)
        min_health = np.min(health_score)
        max_health = np.max(health_score)
        
        # Class distribution
        unique_classes, counts = np.unique(health_classification, return_counts=True)
        total_pixels = health_classification.size
        
        stats_text = f"Health Statistics:\n"
        stats_text += f"Mean Score: {mean_health:.3f}\n"
        stats_text += f"Std Dev: {std_health:.3f}\n"
        stats_text += f"Range: {min_health:.3f} - {max_health:.3f}\n\n"
        stats_text += f"Class Distribution:\n"
        
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            stats_text += f"{cls}: {percentage:.1f}%\n"
        
        return stats_text
    
    def create_combined_health_visualization(self, indices):
        """Create a combined health visualization from your existing approach."""
        # This combines the approach from your existing code
        # Normalize each index to 0-1
        def norm(x):
            x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)
            return np.clip(x, 0, 1)
        
        ndvi_n = norm(indices['ndvi'])
        ndwi_n = norm(indices.get('ndwi', np.zeros_like(indices['ndvi'])))
        savi_n = norm(indices['savi'])
        evi_n = norm(indices['evi'])
        gndvi_n = norm(indices.get('gndvi', np.zeros_like(indices['ndvi'])))
        
        # Weighted combination matching your existing approach
        # You can adjust these weights based on your requirements
        combined = (0.4 * ndvi_n + 0.2 * ndwi_n + 0.2 * savi_n + 
                   0.1 * evi_n + 0.1 * gndvi_n)
        
        return combined
    
    def create_your_style_health_map(self, npy_path, output_path=None):
        """Create health map using your existing visualization approach."""
        # Load image data
        img_data = self.load_image_data(npy_path)
        
        # Calculate indices using your approach
        cube = img_data
        if len(cube.shape) == 4:
            cube = cube[0]  # Remove batch dimension
        if cube.shape[-1] == 1:
            cube = cube[:, :, :, 0]  # Remove channel dimension
        
        # Select bands similar to your approach
        if cube.shape[2] >= 50:  # Hyperspectral case
            red_band = cube[:, :, 30]
            nir_band = cube[:, :, 40]
            swir_band = cube[:, :, 45]
            green_band = cube[:, :, 20]
        else:  # Standard case
            red_band = cube[:, :, min(2, cube.shape[2]-1)]
            nir_band = cube[:, :, min(3, cube.shape[2]-1)]
            swir_band = cube[:, :, min(4, cube.shape[2]-1)] if cube.shape[2] > 4 else cube[:, :, -1]
            green_band = cube[:, :, min(1, cube.shape[2]-1)]
        
        # Calculate indices
        epsilon = 1e-6
        ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
        ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)
        ndsi = (green_band - swir_band) / (green_band + swir_band + epsilon)
        chi = (nir_band - red_band) / (nir_band + red_band + epsilon)  # Same as NDVI
        
        # Combine indices using your approach
        def norm(x):
            x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + epsilon)
            return np.clip(x, 0, 1)
        
        ndvi_n = norm(ndvi)
        ndwi_n = norm(ndwi)
        ndsi_n = norm(ndsi)
        chi_n = norm(chi)
        
        # Weighted combination from your code
        health_map = (0.5 * ndvi_n) + (0.2 * ndwi_n) + (0.2 * ndsi_n) + (0.1 * chi_n)
        
        # Classify zones
        classified = np.zeros_like(health_map)
        classified[health_map >= 0.6] = 2  # Healthy
        classified[(health_map >= 0.3) & (health_map < 0.6)] = 1  # Stressed
        classified[health_map < 0.3] = 0  # Diseased
        
        # Create visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Health map
        im1 = ax[0].imshow(health_map, cmap='YlGn', vmin=0, vmax=1)
        ax[0].set_title("Composite Crop Health Index (0-1)", fontweight='bold')
        ax[0].axis('off')
        plt.colorbar(im1, ax=ax[0])
        
        # Classification map
        cmap = plt.cm.get_cmap('RdYlGn', 3)
        im2 = ax[1].imshow(classified, cmap=cmap, vmin=0, vmax=2)
        ax[1].set_title("Health Classification Map", fontweight='bold')
        ax[1].axis('off')
        cbar = plt.colorbar(im2, ax=ax[1], ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Diseased', 'Stressed', 'Healthy'])
        
        plt.tight_layout()
        
        # Save the map
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            output_path = f"../outputs/your_style_healthmap_{base_name}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Your style health map saved to: {output_path}")
        
        plt.show()
        
        return {
            'health_map': health_map,
            'classified': classified,
            'output_path': output_path
        }
    
    def create_comparison_map(self, npy_paths, output_path=None, titles=None):
        """Create a comparison map for multiple dates/regions."""
        n_images = len(npy_paths)
        fig, axes = plt.subplots(2, n_images, figsize=(6*n_images, 12))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        health_scores = []
        
        for i, npy_path in enumerate(npy_paths):
            # Load and process image
            img_data = self.load_image_data(npy_path)
            indices = self.calculate_vegetation_indices(img_data)
            health_score = self.calculate_health_score(indices)
            health_classification = self.classify_health(health_score)
            
            health_scores.append(health_score)
            
            # Plot RGB image
            if img_data.shape[2] >= 3:
                rgb_img = img_data[:, :, [2, 1, 0]]
                rgb_img = np.clip(rgb_img, 0, 1)
                axes[0, i].imshow(rgb_img)
            
            title = titles[i] if titles and i < len(titles) else f"Image {i+1}"
            axes[0, i].set_title(f"RGB - {title}")
            axes[0, i].axis('off')
            
            # Plot health score
            im = axes[1, i].imshow(health_score, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1, i].set_title(f"Health Score - {title}")
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], shrink=0.8)
        
        plt.tight_layout()
        
        # Save comparison
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"../outputs/health_comparison_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison map saved to: {output_path}")
        
        plt.show()
        
        return health_scores

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Generate crop health maps from .npy files')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='Path to .npy file or directory containing .npy files')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output path for the health map')
    parser.add_argument('--title', '-t', type=str, 
                       help='Title for the health map')
    parser.add_argument('--compare', action='store_true', 
                       help='Create comparison map for multiple files')
    parser.add_argument('--stats', action='store_true', default=True,
                       help='Show statistics on the map')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CropHealthMapGenerator()
    
    if os.path.isfile(args.input):
        # Single file processing
        if args.input.endswith('.npy'):
            result = generator.create_health_map(
                args.input, 
                output_path=args.output,
                title=args.title,
                show_stats=args.stats
            )
            print("Health map generation completed!")
        else:
            print("Error: Input file must be a .npy file")
    
    elif os.path.isdir(args.input):
        # Directory processing
        npy_files = [f for f in os.listdir(args.input) if f.endswith('.npy')]
        npy_paths = [os.path.join(args.input, f) for f in sorted(npy_files)]
        
        if not npy_paths:
            print("No .npy files found in the specified directory")
            return
        
        if args.compare and len(npy_paths) > 1:
            # Create comparison map
            titles = [os.path.splitext(f)[0] for f in npy_files]
            generator.create_comparison_map(npy_paths[:5], args.output, titles[:5])  # Limit to 5 images
        else:
            # Process each file individually
            for npy_path in npy_paths:
                base_name = os.path.splitext(os.path.basename(npy_path))[0]
                output_path = f"../outputs/healthmap_{base_name}.png" if not args.output else args.output
                title = f"Health Map - {base_name}" if not args.title else args.title
                
                generator.create_health_map(
                    npy_path,
                    output_path=output_path,
                    title=title,
                    show_stats=args.stats
                )
    else:
        print("Error: Input path does not exist")

# If run without arguments, process sample data
if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        print("Running with sample data...")
        generator = CropHealthMapGenerator()
        
        # Process first available .npy file
        data_dir = "../data/images"
        if os.path.exists(data_dir):
            npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
            if npy_files:
                sample_file = os.path.join(data_dir, npy_files[0])
                print(f"Processing sample file: {sample_file}")
                generator.create_health_map(sample_file)
                
                # Also create your style visualization for comparison
                print("\nCreating additional health analysis...")
                generator.create_your_style_health_map(sample_file)
            else:
                print("No .npy files found in data/images directory")
        else:
            print("Data directory not found. Please run generate_sample_data.py first.")
    else:
        main()