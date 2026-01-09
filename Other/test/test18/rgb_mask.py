from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image in RGB
img = Image.open("image.png").convert("RGB")
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

# Save individual band images (grayscale)
Image.fromarray(r_thresh.astype(np.uint8), mode='L').save("red_band_grayscale.png")
Image.fromarray(g_thresh.astype(np.uint8), mode='L').save("green_band_grayscale.png")
Image.fromarray(y_thresh.astype(np.uint8), mode='L').save("yellow_band_grayscale.png")

# Save original enhanced bands (without threshold)
Image.fromarray(r_enh, mode='L').save("red_band_original.png")
Image.fromarray(g_enh, mode='L').save("green_band_original.png")
Image.fromarray(y_enh, mode='L').save("yellow_band_original.png")

print("✅ Saved colored threshold images: red_band_threshold.png, green_band_threshold.png, yellow_band_threshold.png")
print("✅ Saved grayscale threshold bands: red_band_grayscale.png, green_band_grayscale.png, yellow_band_grayscale.png")
print("✅ Saved original bands: red_band_original.png, green_band_original.png, yellow_band_original.png")
print(f"🎯 Showing only areas above 75th percentile threshold")

# Step 6: Preview with threshold comparison
plt.figure(figsize=(18, 12))

# Top row: Original enhanced bands
plt.subplot(2, 3, 1)
red_rgb_orig = np.stack([r_enh, np.zeros_like(r_enh), np.zeros_like(r_enh)], axis=-1)
# plt.imshow(red_rgb_orig)
plt.title("🔴 Red Band (Original)")
plt.axis('off')

plt.subplot(2, 3, 2)
green_rgb_orig = np.stack([np.zeros_like(g_enh), g_enh, np.zeros_like(g_enh)], axis=-1)
# plt.imshow(green_rgb_orig)
plt.title("🟢 Green Band (Original)")
plt.axis('off')

plt.subplot(2, 3, 3)
yellow_rgb_orig = np.stack([y_enh, y_enh, np.zeros_like(y_enh)], axis=-1)
plt.imshow(yellow_rgb_orig)
plt.title("🟡 Yellow Band (Original)")
plt.axis('off')

# Bottom row: Thresholded bands (high concentration areas only)
plt.subplot(2, 3, 4)
plt.imshow(red_rgb)
plt.title("🔴 Red Band (High Concentration Only)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(green_rgb)
plt.title("🟢 Green Band (High Concentration Only)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(yellow_rgb)
plt.title("🟡 Yellow Band (High Concentration Only)")
plt.axis('off')

plt.tight_layout()
plt.show()
