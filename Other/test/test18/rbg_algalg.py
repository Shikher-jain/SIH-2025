from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Step 1: Load image in RGB
img = Image.open("image.png")#.convert("RGB")
img_np = np.array(img).astype(np.float32)

# Step 2: Extract Red & Green
r = img_np[:, :, 0]
g = img_np[:, :, 1]

# Step 3: Compute Yellow = Red + Green
yellow = r + g  # could go up to 510

# 🔧 Enhance contrast using percentile stretch
def enhance(img, low=2, high=98):
    p1, p2 = np.percentile(img, (low, high))
    img_stretched = np.clip((img - p1) / (p2 - p1) * 255, 0, 255)
    return img_stretched.astype(np.uint8)

# 🎯 Apply threshold to show only high concentration areas
def apply_threshold(img, threshold_percent=70):
    threshold_value = np.percentile(img, threshold_percent)
    mask = img >= threshold_value
    result = np.where(mask, img, 255)  # Keep original values above threshold, set others to white (255)
    return result, mask

r_enh = enhance(r)
g_enh = enhance(g)
y_enh = enhance(yellow)

# Apply thresholding to show only high concentration areas
r_thresh, r_mask = apply_threshold(r_enh, threshold_percent=55)
g_thresh, g_mask = apply_threshold(g_enh, threshold_percent=55)
y_thresh, y_mask = apply_threshold(y_enh, threshold_percent=55)

# 🎨 Create colorful heatmap masks for each band
def create_colorful_mask(img, mask, colormap_name):
    # Apply colormap to the image
    colormap = cm.get_cmap(colormap_name)
    colored_img = colormap(img / 255.0)  # Normalize to 0-1 for colormap
    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)  # Convert back to RGB
    
    # Create white background where mask is False
    white_bg = np.full_like(colored_img, 255)
    result = np.where(mask[..., np.newaxis], colored_img, white_bg)
    return result

# 🚨 Create crop damage detection mask
def create_damage_mask(red_mask, green_mask, yellow_mask, img_shape):
    """
    Create a comprehensive damage detection mask that clearly shows:
    - Red areas: Severe damage/stress (shown in bright red)
    - Yellow areas: Moderate damage/nutrient deficiency (shown in orange)
    - Green areas: Pest/disease activity (shown in bright green)
    - Combined damage areas (shown in purple)
    """
    # Create damage severity map
    damage_map = np.zeros((*img_shape[:2], 3), dtype=np.uint8)
    
    # Severe damage (Red band high) - Bright Red
    damage_map[red_mask] = [255, 0, 0]  # Bright Red
    
    # Moderate damage (Yellow band high) - Orange
    damage_map[yellow_mask & ~red_mask] = [255, 165, 0]  # Orange
    
    # Pest/Disease (Green band high) - Bright Green  
    damage_map[green_mask & ~red_mask & ~yellow_mask] = [0, 255, 0]  # Bright Green
    
    # Combined damage (multiple bands) - Purple (Critical)
    combined_mask = (red_mask.astype(int) + green_mask.astype(int) + yellow_mask.astype(int)) >= 2
    damage_map[combined_mask] = [128, 0, 128]  # Purple
    
    # Background remains white for healthy areas
    healthy_mask = ~(red_mask | green_mask | yellow_mask)
    damage_map[healthy_mask] = [255, 255, 255]  # White background
    
    return damage_map, combined_mask

# 📊 Create damage statistics overlay
def add_damage_statistics(img_array, red_mask, green_mask, yellow_mask, combined_mask):
    """
    Add text overlay showing damage statistics
    """
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    
    # Calculate damage percentages
    total_pixels = red_mask.size
    red_damage_pct = (np.sum(red_mask) / total_pixels) * 100
    green_damage_pct = (np.sum(green_mask) / total_pixels) * 100
    yellow_damage_pct = (np.sum(yellow_mask) / total_pixels) * 100
    critical_damage_pct = (np.sum(combined_mask) / total_pixels) * 100
    
    # Create damage report text
    stats_text = [
        f"🔴 Severe Damage: {red_damage_pct:.1f}%",
        f"🟠 Moderate Damage: {yellow_damage_pct:.1f}%", 
        f"🟢 Pest/Disease: {green_damage_pct:.1f}%",
        f"🟣 Critical Areas: {critical_damage_pct:.1f}%"
    ]
    
    # Add text overlay (top-left corner)
    y_offset = 10
    for text in stats_text:
        # Add black outline for better visibility
        for adj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            draw.text((10+adj[0], y_offset+adj[1]), text, fill=(0,0,0))
        draw.text((10, y_offset), text, fill=(255,255,255))
        y_offset += 25
    
    return np.array(img_pil)

# 🎯 Create severity zones map
def create_severity_zones(red_mask, green_mask, yellow_mask, img_shape):
    """
    Create zones based on damage severity for easy interpretation
    """
    zones = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # Zone 1: Healthy (no damage) - Value 0 (White)
    # Zone 2: Low concern (single band) - Value 85 (Light Gray)
    # Zone 3: Moderate concern (yellow) - Value 170 (Dark Gray) 
    # Zone 4: High concern (red or multiple bands) - Value 255 (Black)
    
    # Single band detections (low concern)
    single_band = (red_mask.astype(int) + green_mask.astype(int) + yellow_mask.astype(int)) == 1
    zones[single_band] = 85
    
    # Yellow predominant (moderate concern)
    zones[yellow_mask & ~red_mask] = 170
    
    # Red or multiple bands (high concern)
    high_concern = red_mask | ((red_mask.astype(int) + green_mask.astype(int) + yellow_mask.astype(int)) >= 2)
    zones[high_concern] = 255
    
    return zones

# Create eye-catching colorful masks
red_heatmap = create_colorful_mask(r_thresh, r_mask, 'Reds')  # Red heatmap
green_heatmap = create_colorful_mask(g_thresh, g_mask, 'Greens')  # Green heatmap
yellow_heatmap = create_colorful_mask(y_thresh, y_mask, 'plasma')  # Plasma colormap for yellow

# Create additional artistic masks
red_fire = create_colorful_mask(r_thresh, r_mask, 'hot')  # Fire-like
green_nature = create_colorful_mask(g_thresh, g_mask, 'viridis')  # Nature-like
yellow_electric = create_colorful_mask(y_thresh, y_mask, 'inferno')  # Electric-like

# 🚨 CREATE CROP DAMAGE ANALYSIS MASKS
# Create comprehensive damage detection mask
damage_mask, critical_areas = create_damage_mask(r_mask, g_mask, y_mask, img_np.shape)

# Add damage statistics overlay
damage_with_stats = add_damage_statistics(damage_mask, r_mask, g_mask, y_mask, critical_areas)

# Create severity zones map
severity_zones = create_severity_zones(r_mask, g_mask, y_mask, img_np.shape)

# Create colored severity zones for better visualization
colored_zones = np.zeros((*img_np.shape[:2], 3), dtype=np.uint8)
colored_zones[severity_zones == 0] = [255, 255, 255]    # Healthy - White
colored_zones[severity_zones == 85] = [144, 238, 144]   # Low concern - Light Green
colored_zones[severity_zones == 170] = [255, 140, 0]    # Moderate - Orange
colored_zones[severity_zones == 255] = [220, 20, 60]    # High concern - Crimson

# Step 4: Create colored images with thresholding (white background)
red_rgb    = np.stack([r_thresh, np.full_like(r_thresh, 255), np.full_like(r_thresh, 255)], axis=-1)
red_rgb    = np.where(r_mask[..., np.newaxis], np.stack([r_thresh, np.zeros_like(r_thresh), np.zeros_like(r_thresh)], axis=-1), red_rgb)

green_rgb  = np.stack([np.full_like(g_thresh, 255), g_thresh, np.full_like(g_thresh, 255)], axis=-1)
green_rgb  = np.where(g_mask[..., np.newaxis], np.stack([np.zeros_like(g_thresh), g_thresh, np.zeros_like(g_thresh)], axis=-1), green_rgb)

yellow_rgb = np.stack([y_thresh, y_thresh, np.full_like(y_thresh, 255)], axis=-1)
yellow_rgb = np.where(y_mask[..., np.newaxis], np.stack([y_thresh, y_thresh, np.zeros_like(y_thresh)], axis=-1), yellow_rgb)

# Step 5: Save thresholded images showing only high concentration areas
# Save colored versions
Image.fromarray(red_rgb).save("red_band_threshold.png")
Image.fromarray(green_rgb).save("green_band_threshold.png")
Image.fromarray(yellow_rgb).save("yellow_band_threshold.png")

# 🎨 Save colorful heatmap masks
Image.fromarray(red_heatmap).save("red_heatmap_mask.png")
Image.fromarray(green_heatmap).save("green_heatmap_mask.png")
Image.fromarray(yellow_heatmap).save("yellow_heatmap_mask.png")

# 🔥 Save artistic style masks
Image.fromarray(red_fire).save("red_fire_mask.png")
Image.fromarray(green_nature).save("green_nature_mask.png")
Image.fromarray(yellow_electric).save("yellow_electric_mask.png")

# 🚨 SAVE CROP DAMAGE ANALYSIS RESULTS
# Save main damage detection masks
Image.fromarray(damage_mask).save("CROP_DAMAGE_MASK.png")
Image.fromarray(damage_with_stats).save("DAMAGE_ANALYSIS_WITH_STATS.png")
Image.fromarray(colored_zones).save("DAMAGE_SEVERITY_ZONES.png")
Image.fromarray(severity_zones, mode='L').save("severity_zones_grayscale.png")

# Save individual band images (grayscale)
Image.fromarray(r_thresh.astype(np.uint8), mode='L').save("red_band_grayscale.png")
Image.fromarray(g_thresh.astype(np.uint8), mode='L').save("green_band_grayscale.png")
Image.fromarray(y_thresh.astype(np.uint8), mode='L').save("yellow_band_grayscale.png")

# Save original enhanced bands (without threshold)
Image.fromarray(r_enh, mode='L').save("red_band_original.png")
Image.fromarray(g_enh, mode='L').save("green_band_original.png")
Image.fromarray(y_enh, mode='L').save("yellow_band_original.png")

print("✅ Saved colored threshold images: red_band_threshold.png, green_band_threshold.png, yellow_band_threshold.png")
print("🎨 Saved heatmap masks: red_heatmap_mask.png, green_heatmap_mask.png, yellow_heatmap_mask.png")
print("🔥 Saved artistic masks: red_fire_mask.png, green_nature_mask.png, yellow_electric_mask.png")
print("")
print("🚨 === CROP DAMAGE ANALYSIS RESULTS ===")
print("📊 CROP_DAMAGE_MASK.png - Shows all damage types in different colors")
print("📈 DAMAGE_ANALYSIS_WITH_STATS.png - Includes damage statistics overlay")
print("🎯 DAMAGE_SEVERITY_ZONES.png - Color-coded severity zones for quick assessment")
print("")
print("🔴 Red = Severe crop damage/stress")
print("🟠 Orange = Moderate damage/nutrient deficiency")
print("🟢 Green = Pest/disease activity")
print("🟣 Purple = Critical areas (multiple damage types)")
print("⚪ White = Healthy crop areas")
print("")
print("✅ Saved grayscale threshold bands: red_band_grayscale.png, green_band_grayscale.png, yellow_band_grayscale.png")
print("✅ Saved original bands: red_band_original.png, green_band_original.png, yellow_band_original.png")
print(f"🎯 Showing only areas above 55th percentile threshold")

# Print damage statistics
total_pixels = r_mask.size
red_damage_pct = (np.sum(r_mask) / total_pixels) * 100
green_damage_pct = (np.sum(g_mask) / total_pixels) * 100
yellow_damage_pct = (np.sum(y_mask) / total_pixels) * 100
critical_damage_pct = (np.sum(critical_areas) / total_pixels) * 100

print("\n📊 DAMAGE ASSESSMENT SUMMARY:")
print(f"🔴 Severe Damage Areas: {red_damage_pct:.1f}% of total crop")
print(f"🟠 Moderate Damage Areas: {yellow_damage_pct:.1f}% of total crop")
print(f"🟢 Pest/Disease Areas: {green_damage_pct:.1f}% of total crop")
print(f"🟣 Critical Damage Areas: {critical_damage_pct:.1f}% of total crop")
print(f"⚪ Healthy Crop Areas: {100 - (red_damage_pct + green_damage_pct + yellow_damage_pct - critical_damage_pct):.1f}% of total crop")

# Step 6: Preview with crop damage analysis focus
plt.figure(figsize=(24, 18))

# Row 1: Original image and main damage detection
plt.subplot(3, 4, 1)
plt.imshow(img_np.astype(np.uint8))
plt.title("📷 Original Image")
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(damage_mask)
plt.title("🚨 CROP DAMAGE DETECTION\n(Main Analysis)")
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(damage_with_stats)
plt.title("📊 DAMAGE WITH STATISTICS\n(Includes % Data)")
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(colored_zones)
plt.title("🎯 SEVERITY ZONES\n(Quick Assessment)")
plt.axis('off')

# Row 2: Individual band analysis
plt.subplot(3, 4, 5)
plt.imshow(red_rgb)
plt.title("🔴 SEVERE DAMAGE\n(Red Band Threshold)")
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(yellow_rgb)
plt.title("🟠 MODERATE DAMAGE\n(Yellow Band Threshold)")
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(green_rgb)
plt.title("🟢 PEST/DISEASE\n(Green Band Threshold)")
plt.axis('off')

plt.subplot(3, 4, 8)
critical_display = np.zeros_like(damage_mask)
critical_display[critical_areas] = [128, 0, 128]  # Purple for critical
critical_display[~critical_areas] = [255, 255, 255]  # White background
plt.imshow(critical_display)
plt.title("🟣 CRITICAL AREAS\n(Multiple Damage Types)")
plt.axis('off')

# Row 3: Heatmap style masks for detailed analysis
plt.subplot(3, 4, 9)
plt.imshow(red_heatmap)
plt.title("🌶️ Red Intensity Heatmap")
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(green_heatmap)
plt.title("🌿 Green Intensity Heatmap")
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(yellow_heatmap)
plt.title("⚡ Yellow Intensity Heatmap")
plt.axis('off')

plt.subplot(3, 4, 12)
# Create a legend/summary plot
legend_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
plt.imshow(legend_img)
plt.title("📋 DAMAGE LEGEND")
legend_text = [
    "🔴 Red = Severe Damage",
    "🟠 Orange = Moderate", 
    "🟢 Green = Pest/Disease",
    "🟣 Purple = Critical",
    "⚪ White = Healthy"
]
plt.text(10, 150, '\n'.join(legend_text), fontsize=10, verticalalignment='top')
plt.axis('off')

plt.suptitle("🚨 COMPREHENSIVE CROP DAMAGE ANALYSIS 🚨", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
