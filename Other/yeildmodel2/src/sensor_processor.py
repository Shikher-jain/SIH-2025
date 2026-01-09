import numpy as np
from flask import jsonify

def generate_yield_mask(yield_data, thresholds=(0.3, 0.6)):
    """Generate a red-yellow-green mask based on yield values."""
    try:
        # Ensure yield_data is a numpy array
        if not isinstance(yield_data, np.ndarray):
            raise ValueError("Input yield_data must be a numpy array.")
        
        # Create an empty mask with the same shape as yield_data
        mask = np.zeros(yield_data.shape, dtype=np.uint8)

        # Apply thresholds to create the mask
        mask[yield_data < thresholds[0]] = 0  # Red for low yield
        mask[(yield_data >= thresholds[0]) & (yield_data < thresholds[1])] = 128  # Yellow for moderate yield
        mask[yield_data >= thresholds[1]] = 255  # Green for high yield

        return mask
    except Exception as e:
        print(f"Error generating yield mask: {e}")
        return None

def process_sensor_data(sensor_data, yield_thresholds=(0.3, 0.6)):
    """Process sensor data and generate a yield mask."""
    try:
        # Assuming sensor_data is a numpy array of yield values
        yield_mask = generate_yield_mask(sensor_data, thresholds=yield_thresholds)
        
        if yield_mask is not None:
            return jsonify({
                "status": "success",
                "mask": yield_mask.tolist()  # Convert to list for JSON serialization
            })
        else:
            return jsonify({"status": "error", "message": "Failed to generate yield mask."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500