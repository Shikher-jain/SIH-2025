#!/usr/bin/env python3
"""
Color Separator Script - Extract red, yellow, and green regions from crop health NDVI images
Analyzes crop_health_ndvi_256.png files and creates separate images for each health zone
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def analyze_image_colors(image_path):
    """
    Analyze the color distribution in an NDVI crop health image
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        dict: Color analysis results
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Get image dimensions
            height, width = img_array.shape[:2]
            total_pixels = height * width
            
            # Initialize masks
            red_mask = np.zeros((height, width), dtype=bool)
            yellow_mask = np.zeros((height, width), dtype=bool)
            green_mask = np.zeros((height, width), dtype=bool)
            
            # Convert RGB to HSV for better color separation
            from colorsys import rgb_to_hsv
            
            # Analyze each pixel using HSV color space
            for y in range(height):
                for x in range(width):
                    r, g, b = img_array[y, x]
                    
                    # Skip very dark pixels (likely background)
                    if r < 30 and g < 30 and b < 30:
                        continue
                    
                    # Convert to HSV (0-1 range)
                    h, s, v = rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                    h_deg = h * 360  # Convert to degrees
                    
                    # Classify based on hue and saturation
                    if s > 0.3 and v > 0.2:  # Ignore low saturation/brightness
                        if h_deg >= 340 or h_deg <= 30:  # Red hue range
                            red_mask[y, x] = True
                        elif 40 <= h_deg <= 80:  # Yellow hue range
                            yellow_mask[y, x] = True
                        elif 90 <= h_deg <= 150:  # Green hue range
                            green_mask[y, x] = True
                        else:
                            # For other colors, classify by dominant RGB channel
                            if r > g and r > b and r > 100:
                                red_mask[y, x] = True
                            elif g > r and g > b and g > 100:
                                green_mask[y, x] = True
                            elif r > 120 and g > 120 and b < 100:
                                yellow_mask[y, x] = True
            
            # Calculate statistics
            red_pixels = np.sum(red_mask)
            yellow_pixels = np.sum(yellow_mask)
            green_pixels = np.sum(green_mask)
            other_pixels = total_pixels - (red_pixels + yellow_pixels + green_pixels)
            
            results = {
                'total_pixels': total_pixels,
                'red_pixels': red_pixels,
                'yellow_pixels': yellow_pixels,
                'green_pixels': green_pixels,
                'other_pixels': other_pixels,
                'red_percentage': (red_pixels / total_pixels) * 100,
                'yellow_percentage': (yellow_pixels / total_pixels) * 100,
                'green_percentage': (green_pixels / total_pixels) * 100,
                'other_percentage': (other_pixels / total_pixels) * 100,
                'red_mask': red_mask,
                'yellow_mask': yellow_mask,
                'green_mask': green_mask,
                'original_image': img_array
            }
            
            return results
            
    except Exception as e:
        print(f"❌ Error analyzing image {image_path}: {e}")
        return None

def create_color_separated_images(analysis_results, output_dir, base_name):
    """
    Create separate images for red, yellow, and green zones with white backgrounds
    showing original color variations within each zone
    
    Args:
        analysis_results (dict): Results from analyze_image_colors
        output_dir (str): Directory to save output images
        base_name (str): Base name for output files
    """
    try:
        original = analysis_results['original_image']
        height, width = original.shape[:2]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create white background (255, 255, 255)
        white_bg = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # Create red zone image (poor crop health) - preserve original colors
        red_image = white_bg.copy()
        red_mask = analysis_results['red_mask']
        if np.any(red_mask):
            # Keep original colors but enhance red tones
            red_image[red_mask] = original[red_mask]
            # Enhance red channel slightly for better visibility
            red_image[red_mask, 0] = np.clip(red_image[red_mask, 0] * 1.2, 0, 255)
        
        red_pil = Image.fromarray(red_image)
        red_path = os.path.join(output_dir, f"{base_name}_red_zone.png")
        red_pil.save(red_path)
        print(f"✅ Saved red zone (poor health): {red_path}")
        
        # Create yellow zone image (moderate crop health) - preserve original colors
        yellow_image = white_bg.copy()
        yellow_mask = analysis_results['yellow_mask']
        if np.any(yellow_mask):
            # Keep original colors but enhance yellow tones
            yellow_image[yellow_mask] = original[yellow_mask]
            # Enhance yellow (red + green) channels slightly
            yellow_image[yellow_mask, 0] = np.clip(yellow_image[yellow_mask, 0] * 1.1, 0, 255)
            yellow_image[yellow_mask, 1] = np.clip(yellow_image[yellow_mask, 1] * 1.1, 0, 255)
        
        yellow_pil = Image.fromarray(yellow_image)
        yellow_path = os.path.join(output_dir, f"{base_name}_yellow_zone.png")
        yellow_pil.save(yellow_path)
        print(f"✅ Saved yellow zone (moderate health): {yellow_path}")
        
        # Create green zone image (good crop health) - preserve original colors
        green_image = white_bg.copy()
        green_mask = analysis_results['green_mask']
        if np.any(green_mask):
            # Keep original colors but enhance green tones
            green_image[green_mask] = original[green_mask]
            # Enhance green channel slightly for better visibility
            green_image[green_mask, 1] = np.clip(green_image[green_mask, 1] * 1.2, 0, 255)
        
        green_pil = Image.fromarray(green_image)
        green_path = os.path.join(output_dir, f"{base_name}_green_zone.png")
        green_pil.save(green_path)
        print(f"✅ Saved green zone (good health): {green_path}")
        
        # Create a composite overview image
        create_composite_overview(analysis_results, output_dir, base_name)
        
        # Also create pure color versions for comparison
        create_pure_color_versions(analysis_results, output_dir, base_name)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating separated images: {e}")
        return False

def create_pure_color_versions(analysis_results, output_dir, base_name):
    """
    Create pure color versions (solid colors) for clear zone identification
    """
    try:
        height, width = analysis_results['original_image'].shape[:2]
        white_bg = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # Pure red zones
        pure_red_image = white_bg.copy()
        red_mask = analysis_results['red_mask']
        if np.any(red_mask):
            pure_red_image[red_mask] = [255, 0, 0]
        
        red_pil = Image.fromarray(pure_red_image)
        red_path = os.path.join(output_dir, f"{base_name}_pure_red_zone.png")
        red_pil.save(red_path)
        
        # Pure yellow zones
        pure_yellow_image = white_bg.copy()
        yellow_mask = analysis_results['yellow_mask']
        if np.any(yellow_mask):
            pure_yellow_image[yellow_mask] = [255, 255, 0]
        
        yellow_pil = Image.fromarray(pure_yellow_image)
        yellow_path = os.path.join(output_dir, f"{base_name}_pure_yellow_zone.png")
        yellow_pil.save(yellow_path)
        
        # Pure green zones
        pure_green_image = white_bg.copy()
        green_mask = analysis_results['green_mask']
        if np.any(green_mask):
            pure_green_image[green_mask] = [0, 255, 0]
        
        green_pil = Image.fromarray(pure_green_image)
        green_path = os.path.join(output_dir, f"{base_name}_pure_green_zone.png")
        green_pil.save(green_path)
        
        print(f"💡 Also saved pure color versions with '_pure_' prefix")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating pure color versions: {e}")
        return False

def create_composite_overview(analysis_results, output_dir, base_name):
    """
    Create a composite overview showing all zones and statistics
    """
    try:
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Crop Health Analysis - {base_name}', fontsize=16, fontweight='bold')
        
        original = analysis_results['original_image']
        red_mask = analysis_results['red_mask']
        yellow_mask = analysis_results['yellow_mask']
        green_mask = analysis_results['green_mask']
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original NDVI Image')
        axes[0, 0].axis('off')
        
        # Red zones (poor health)
        red_overlay = np.zeros_like(original)
        red_overlay[red_mask] = [255, 0, 0]  # Pure red
        axes[0, 1].imshow(red_overlay)
        axes[0, 1].set_title(f'Poor Health Zones\n({analysis_results["red_percentage"]:.1f}%)')
        axes[0, 1].axis('off')
        
        # Yellow zones (moderate health)
        yellow_overlay = np.zeros_like(original)
        yellow_overlay[yellow_mask] = [255, 255, 0]  # Pure yellow
        axes[0, 2].imshow(yellow_overlay)
        axes[0, 2].set_title(f'Moderate Health Zones\n({analysis_results["yellow_percentage"]:.1f}%)')
        axes[0, 2].axis('off')
        
        # Green zones (good health)
        green_overlay = np.zeros_like(original)
        green_overlay[green_mask] = [0, 255, 0]  # Pure green
        axes[1, 0].imshow(green_overlay)
        axes[1, 0].set_title(f'Good Health Zones\n({analysis_results["green_percentage"]:.1f}%)')
        axes[1, 0].axis('off')
        
        # Combined zones
        combined = np.zeros_like(original)
        combined[red_mask] = [255, 100, 100]    # Light red
        combined[yellow_mask] = [255, 255, 100] # Light yellow
        combined[green_mask] = [100, 255, 100]  # Light green
        axes[1, 1].imshow(combined)
        axes[1, 1].set_title('All Health Zones Combined')
        axes[1, 1].axis('off')
        
        # Statistics pie chart
        labels = ['Poor (Red)', 'Moderate (Yellow)', 'Good (Green)', 'Other']
        sizes = [
            analysis_results['red_percentage'],
            analysis_results['yellow_percentage'], 
            analysis_results['green_percentage'],
            analysis_results['other_percentage']
        ]
        colors = ['#ff4444', '#ffff44', '#44ff44', '#cccccc']
        
        # Only include non-zero segments
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0.1]
        filtered_labels = [labels[i] for i in non_zero_indices]
        filtered_sizes = [sizes[i] for i in non_zero_indices]
        filtered_colors = [colors[i] for i in non_zero_indices]
        
        axes[1, 2].pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Health Distribution')
        
        plt.tight_layout()
        
        # Save composite image
        composite_path = os.path.join(output_dir, f"{base_name}_analysis_overview.png")
        plt.savefig(composite_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved analysis overview: {composite_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating composite overview: {e}")
        return False

def process_single_file(input_path, output_dir=None):
    """
    Process a single crop_health_ndvi_256.png file
    
    Args:
        input_path (str): Path to input image
        output_dir (str): Output directory (if None, creates subdirectory next to input)
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"❌ Input file does not exist: {input_path}")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = input_file.parent / "color_separated"
    else:
        output_dir = Path(output_dir)
    
    # Get base name without extension
    base_name = input_file.stem
    
    print(f"Processing: {input_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Analyze image colors
    analysis = analyze_image_colors(str(input_file))
    if analysis is None:
        return False
    
    # Print statistics
    print(f"📊 Color Analysis Results:")
    print(f"  Total pixels: {analysis['total_pixels']:,}")
    print(f"  Red (Poor health): {analysis['red_pixels']:,} pixels ({analysis['red_percentage']:.1f}%)")
    print(f"  Yellow (Moderate health): {analysis['yellow_pixels']:,} pixels ({analysis['yellow_percentage']:.1f}%)")
    print(f"  Green (Good health): {analysis['green_pixels']:,} pixels ({analysis['green_percentage']:.1f}%)")
    print(f"  Other colors: {analysis['other_pixels']:,} pixels ({analysis['other_percentage']:.1f}%)")
    print()
    
    # Create separated images
    success = create_color_separated_images(analysis, str(output_dir), base_name)
    
    if success:
        print(f"✅ Successfully processed {input_path}")
        print(f"📁 Output files saved in: {output_dir}")
    else:
        print(f"❌ Failed to process {input_path}")
    
    return success

def process_directory_recursive(root_dir, output_base_dir=None):
    """
    Recursively process all crop_health_ndvi_256.png files in a directory
    
    Args:
        root_dir (str): Root directory to search
        output_base_dir (str): Base output directory
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ Directory does not exist: {root_dir}")
        return False
    
    # Find all crop_health_ndvi_256.png files
    pattern = "**/crop_health_ndvi_256.png"
    ndvi_files = list(root_path.rglob(pattern))
    
    if not ndvi_files:
        print(f"❌ No crop_health_ndvi_256.png files found in {root_dir}")
        return False
    
    print(f"Found {len(ndvi_files)} NDVI crop health files to process")
    print("=" * 60)
    
    success_count = 0
    
    for ndvi_file in ndvi_files:
        # Create output directory relative to input
        if output_base_dir:
            relative_path = ndvi_file.parent.relative_to(root_path)
            output_dir = Path(output_base_dir) / relative_path / "color_separated"
        else:
            output_dir = ndvi_file.parent / "color_separated"
        
        if process_single_file(str(ndvi_file), str(output_dir)):
            success_count += 1
        
        print()  # Add spacing between files
    
    print("=" * 60)
    print(f"✅ Successfully processed {success_count}/{len(ndvi_files)} files")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Separate crop health NDVI images into red, yellow, and green zones')
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('-o', '--output', help='Output directory path')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Process subdirectories recursively (for directory input)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    print("🎨 Crop Health Color Separator")
    print("=" * 40)
    
    if input_path.is_file():
        # Single file processing
        if "crop_health_ndvi" not in input_path.name:
            print("⚠️  Warning: Input file doesn't appear to be a crop health NDVI image")
            print("    Expected filename pattern: '*crop_health_ndvi*.png'")
            print("    Proceeding anyway...")
        
        process_single_file(str(input_path), args.output)
        
    elif input_path.is_dir():
        # Directory processing
        if args.recursive:
            process_directory_recursive(str(input_path), args.output)
        else:
            # Look for files only in the specified directory
            pattern = "crop_health_ndvi_256.png"
            ndvi_files = list(input_path.glob(pattern))
            
            if not ndvi_files:
                print(f"❌ No {pattern} files found in {input_path}")
                print("💡 Tip: Use -r flag to search subdirectories recursively")
                sys.exit(1)
            
            for ndvi_file in ndvi_files:
                output_dir = args.output if args.output else str(ndvi_file.parent / "color_separated")
                process_single_file(str(ndvi_file), output_dir)
    else:
        print(f"❌ Invalid input path: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    # If no command line arguments, provide interactive mode
    if len(sys.argv) == 1:
        print("🎨 Crop Health Color Separator - Interactive Mode")
        print("=" * 50)
        
        # Get input path
        input_path = input("Enter path to crop_health_ndvi_256.png file or directory: ").strip()
        if not input_path:
            print("❌ No input path provided")
            sys.exit(1)
        
        path = Path(input_path)
        if not path.exists():
            print(f"❌ Path does not exist: {input_path}")
            sys.exit(1)
        
        if path.is_file():
            process_single_file(str(path))
        else:
            recursive = input("Process subdirectories recursively? (y/n) [default: y]: ").strip().lower()
            if recursive == 'n':
                # Look in specified directory only
                pattern = "crop_health_ndvi_256.png"
                ndvi_files = list(path.glob(pattern))
                
                if not ndvi_files:
                    print(f"❌ No {pattern} files found in {path}")
                    sys.exit(1)
                
                for ndvi_file in ndvi_files:
                    process_single_file(str(ndvi_file))
            else:
                process_directory_recursive(str(path))
    else:
        main()