def generate_advisory_report(cell_id, label, indices_dict, weather_dict):
    label_map = {0: 'Stressed', 1: 'Moderate', 2: 'Healthy'}
    report = f"Advisory Report for {cell_id}:\n"
    report += f"1. Crop condition: {label_map.get(label, 'Unknown')}\n"
    # rules
    if indices_dict.get('NDVI_mean',0) < 0.35:
        report += "2. Low vegetation detected — irrigation recommended.\n"
    if weather_dict.get('humidity',0) > 80:
        report += "3. High humidity — monitor for fungal disease.\n"
    if weather_dict.get('precipitation',0) > 10:
        report += "4. Heavy rain expected — delay spraying.\n"
    report += "5. Monitor daily and log field observations.\n"
    return report
