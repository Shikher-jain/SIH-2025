# infer.py
import os
import sys
import numpy as np
import torch
import cv2
from model import UNetMultiHead
from utils import save_heatmap, save_heatmap_advanced, save_focused_heatmap
import argparse

def load_npy_normalize(path):
    # Load structured array and convert to 21-band format
    data_struct = np.load(path)
    
    # Extract all band data and stack to create (21, H, W)
    bands = []
    for field_name in data_struct.dtype.names:
        band_data = data_struct[field_name].astype(np.float32)
        bands.append(band_data)
    
    arr = np.stack(bands, axis=0)  # Shape: (21, H, W)
    
    # simple per-band minmax normalization (same as train)
    for i in range(arr.shape[0]):
        mi, ma = float(arr[i].min()), float(arr[i].max())
        if ma > mi:
            arr[i] = (arr[i] - mi) / (ma - mi)
    return arr

def run_inference(npy_path, model_path="models/best.pth", out_prefix=None, target_size=256, debug=False):
    if out_prefix is None:
        base = os.path.splitext(os.path.basename(npy_path))[0]
        out_prefix = base

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetMultiHead(in_ch=21, base=32).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    img = load_npy_normalize(npy_path)  # (21,H,W)
    b, h, w = img.shape
    if (h, w) != (target_size, target_size):
        img_resized = np.zeros((b, target_size, target_size), dtype=np.float32)
        for i in range(b):
            img_resized[i] = cv2.resize(img[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = img_resized

    tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # 1,21,H,W
    with torch.no_grad():
        pred_mask, pred_yield = model(tensor)
        pred_mask = pred_mask.squeeze(0).cpu().numpy()  # (1,H,W)
        pred_yield = float(pred_yield.squeeze().cpu().item())

    # Debug information
    if debug:
        print(f"Input image shape: {img.shape}")
        print(f"Input value range: [{img.min():.4f}, {img.max():.4f}]")
        print(f"Prediction mask shape: {pred_mask.shape}")
        print(f"Prediction mask range: [{pred_mask.min():.4f}, {pred_mask.max():.4f}]")
        print(f"Prediction mask mean: {pred_mask.mean():.4f}")
        print(f"Prediction mask std: {pred_mask.std():.4f}")
        print(f"Non-zero pixels: {np.count_nonzero(pred_mask > 0.1)} / {pred_mask.size}")

    # save
    os.makedirs("out", exist_ok=True)
    heatmap_path = os.path.join("out", f"{out_prefix}_heatmap.png")
    
    # Create focused heatmap as primary output
    save_focused_heatmap(pred_mask[0], heatmap_path, 
                        predicted_yield=pred_yield, 
                        sample_id=out_prefix)
    
    # Also create advanced analysis if debug mode
    if debug:
        advanced_path = os.path.join("out", f"{out_prefix}_analysis.png")
        save_heatmap_advanced(pred_mask[0], advanced_path, 
                             predicted_yield=pred_yield, 
                             sample_id=out_prefix)
    
    np.save(os.path.join("out", f"{out_prefix}_heatmap.npy"), pred_mask[0])
    with open(os.path.join("out", f"{out_prefix}_yield.txt"), "w") as f:
        f.write(str(pred_yield))

    print("Saved heatmap ->", heatmap_path)
    print("Predicted yield ->", pred_yield)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy", help=".npy input file (21,H,W)")
    parser.add_argument("--model", default="models/best.pth")
    parser.add_argument("--out", default=None, help="output prefix (optional)")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()
    run_inference(args.npy, model_path=args.model, out_prefix=args.out, debug=args.debug)
