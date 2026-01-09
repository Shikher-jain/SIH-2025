def generate_report(predicted_yield, heatmap):
    mean_intensity = heatmap.mean()
    report_parts = []

    if predicted_yield >= 3.5:
        report_parts.append("The expected yield is high.")
    elif predicted_yield >= 2.0:
        report_parts.append("The yield is moderate.")
    else:
        report_parts.append("Low yield predicted for this region.")

    if mean_intensity > 0.6:
        report_parts.append("Vegetation appears dense.")
    elif mean_intensity > 0.3:
        report_parts.append("Vegetation is patchy.")
    else:
        report_parts.append("Sparse vegetation detected.")

    return " ".join(report_parts)
