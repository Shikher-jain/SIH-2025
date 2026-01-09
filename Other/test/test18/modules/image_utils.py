import numpy as np

def crop_and_pad(image, target_height=128, target_width=128, pad_value=0):
    """
    Crop and pad image to required shape.
    
    Args:
        image: Input image array (H, W, C) or (H, W)
        target_height: Target height in pixels
        target_width: Target width in pixels  
        pad_value: Value to use for padding (default: 0 for black)
    
    Returns:
        Cropped and padded image with shape (target_height, target_width, C) or (target_height, target_width)
    """
    if len(image.shape) == 2:
        # Handle grayscale images
        h, w = image.shape
        channels = 1
        image = np.expand_dims(image, axis=2)
    else:
        h, w, channels = image.shape
    
    # Initialize output array with padding value
    output = np.full((target_height, target_width, channels), pad_value, dtype=image.dtype)
    
    # Calculate crop region (center crop)
    start_h = max(0, (h - target_height) // 2)
    start_w = max(0, (w - target_width) // 2)
    end_h = start_h + min(target_height, h)
    end_w = start_w + min(target_width, w)
    
    # Calculate insert position in output (center placement)
    insert_h_start = max(0, (target_height - h) // 2)
    insert_w_start = max(0, (target_width - w) // 2)
    
    # Extract cropped region
    cropped = image[start_h:end_h, start_w:end_w, :]
    
    # Insert cropped region into output
    output[insert_h_start:insert_h_start+cropped.shape[0], 
           insert_w_start:insert_w_start+cropped.shape[1], 
           :] = cropped
    
    # Remove channel dimension if input was 2D
    if channels == 1:
        output = np.squeeze(output, axis=2)
        
    return output