def generate_advisory_report(cell_id, pred_label, indices_dict, weather_dict):
    label_map = {0:"Stressed",1:"Moderate",2:"Healthy"}
    report = f"Advisory Report for {cell_id}:\n"
    report += f"1. Crop condition is {label_map[pred_label]}.\n"

    # Water stress
    if indices_dict.get("NDWI_mean",0)<0.3:
        report += "2. Irrigation required soon (low water content).\n"

    # Nutrient stress
    if indices_dict.get("NDVI_mean",0)<0.4 or indices_dict.get("GNDVI_mean",0)<0.4:
        report += "3. Nutrient supplementation recommended.\n"

    # Pest/Disease risk (placeholder based on humidity)
    if weather_dict["humidity"]>75:
        report += "4. High risk of fungal disease due to high humidity.\n"

    # Weather alert
    if weather_dict["precipitation"]>10:
        report += "5. Heavy rainfall expected. Adjust pesticide application.\n"

    report += "6. Immediate monitoring advised.\n"
    return report
