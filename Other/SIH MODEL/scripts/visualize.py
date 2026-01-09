import numpy as np
import cv2
import matplotlib.pyplot as plt

def overlay_heatmap(gray_image, heatmap, alpha=0.6):
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    base_img = (gray_image * 255).astype(np.uint8)
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(color_map, alpha, base_img, 1 - alpha, 0)
    return overlay

def show_heatmap(hm, title="Heatmap"):
    plt.imshow(hm, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()
