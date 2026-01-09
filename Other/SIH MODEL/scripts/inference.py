import numpy as np
import cv2
from tensorflow.keras.models import load_model
from report_generator import generate_report
from nlp_refiner import refine_report

def predict_one(model_path, npy_path, out_prefix="output"):
    model = load_model(model_path)
    img = np.load(npy_path)
    inp = np.expand_dims(img, axis=0)

    heatmap_pred, yield_pred = model.predict(inp)
    heatmap_pred = heatmap_pred[0, :, :, 0]
    yield_pred = yield_pred[0]

    # Save heatmap
    heatmap_img = (heatmap_pred * 255).astype(np.uint8)
    cv2.imwrite(f"{out_prefix}_heatmap.png", heatmap_img)

    # Generate report
    report = generate_report(yield_pred, heatmap_pred)
    enhanced = refine_report(report)

    print("Basic:", report)
    print("Enhanced:", enhanced)

    with open(f"{out_prefix}_report.txt", "w") as f:
        f.write(enhanced)

    return heatmap_pred, yield_pred

# Example usage:

# predict_one("best_model.h5", "../data/data/A001.npy", out_prefix="A001")
