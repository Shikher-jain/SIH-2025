def generate_advisory_report(cell_id, pred_label, indices, weather):
    labels = {0:"Stressed",1:"Moderate",2:"Healthy"}
    rep = f"Advisory for {cell_id}:\n"
    rep += f"Crop Condition: {labels[pred_label]}\n"
    rep += f"NDVI: {indices.get('NDVI_mean',0):.2f}\n"
    rep += f"Humidity: {weather['humidity']:.1f}%\n"
    rep += f"Temp: {weather['temperature']:.1f} C\n"
    # irrigation advice
    if weather['humidity']<40:
        rep += "Irrigation needed soon.\n"
    else:
        rep += "Irrigation adequate.\n"
    # nutrients
    for nut in ['N','P','ECe','OC','pH']:
        rep += f"{nut}: {indices.get(f'{nut}_mean',0):.2f}\n"
    return rep
