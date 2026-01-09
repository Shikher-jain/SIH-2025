# infer_health.py
import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import argparse
from health_model import UNetHealthMapOnly
from health_utils import save_health_map_png

def load_health_model(model_path, device='cpu'):
    """Load trained health map model."""
    checkpoint = torch.load(model_path, map_location=device)
    model = UNetHealthMapOnly(in_ch=21, base=32)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    return model

def preprocess_npy_data(npy_path, img_size=256):
    """Preprocess NPY satellite data for inference."""
    # Load structured array
    data_struct = np.load(npy_path)
    
    # Extract all band data and stack to create (21, H, W)
    bands = []
    for field_name in data_struct.dtype.names:
        band_data = data_struct[field_name].astype(np.float32)
        bands.append(band_data)
    
    img = np.stack(bands, axis=0)  # Shape: (21, H, W)
    
    # Normalize per-band (per-image minmax)
    for i in range(img.shape[0]):
        band = img[i]
        mi, ma = float(band.min()), float(band.max())
        if ma > mi:
            img[i] = (band - mi) / (ma - mi)
        else:
            img[i] = band  # constant band -> keep as is
    
    # Resize if needed
    if img.shape[1:] != (img_size, img_size):
        img_resized = np.zeros((img.shape[0], img_size, img_size), dtype=np.float32)
        for i in range(img.shape[0]):
            img_resized[i] = cv2.resize(img[i], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = img_resized
    
    # Add batch dimension and convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 21, H, W)
    return img_tensor

def run_health_inference(model_path, input_npy, output_png, device='cpu'):
    """Run health map inference on satellite data."""
    print(f"Loading model from: {model_path}")
    model = load_health_model(model_path, device)
    
    print(f"Loading satellite data from: {input_npy}")
    input_tensor = preprocess_npy_data(input_npy).to(device)
    
    print(f"Running inference...")
    with torch.no_grad():
        pred_health = model(input_tensor)  # (1, 3, H, W)
    
    # Remove batch dimension
    pred_health = pred_health.squeeze(0)  # (3, H, W)
    
    print(f"Saving health map to: {output_png}")
    save_health_map_png(pred_health, output_png)
    
    print(f"✅ Health map inference completed!")
    return output_png

def main():
    parser = argparse.ArgumentParser(description='Generate health map from satellite data')
    parser.add_argument('input_npy', help='Input .npy file containing satellite data')
    parser.add_argument('--model', default='health_models/best_health_model.pth', help='Path to trained model')
    parser.add_argument('--output', help='Output .png file (optional, will auto-generate if not provided)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_npy):
        print(f"Error: Input file '{args.input_npy}' not found.")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return 1
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = os.path.splitext(args.input_npy)[0]
        args.output = f"{input_path}_health_prediction.png"
    
    try:
        run_health_inference(args.model, args.input_npy, args.output, args.device)
        return 0
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        # Test with sample data if available
        sample_file = "data/data/Agra1.npy"
        if os.path.exists(sample_file) and os.path.exists("health_models/best_health_model.pth"):
            print("Running inference on sample data...")
            run_health_inference(
                "health_models/best_health_model.pth", 
                sample_file, 
                "sample_health_prediction.png"
            )
        else:
            print("Usage: python infer_health.py <input.npy> [--model <model.pth>] [--output <output.png>]")
    else:
        exit(main())