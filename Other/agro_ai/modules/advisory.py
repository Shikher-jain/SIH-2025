# modules/advisory.py
import os
from datetime import datetime

OUTPUT = os.path.join(os.getcwd(), "outputs")
os.makedirs(os.path.join(OUTPUT, "reports"), exist_ok=True)

def irrigation_advice(soil_moisture, forecast_precip_mm, leaf_wetness_status):
    if leaf_wetness_status == "Wet":
        return "No irrigation now (leaf wet). Monitor soil moisture."
    if forecast_precip_mm > 10:
        return "Delay irrigation — significant precipitation forecast."
    if soil_moisture < 25:
        return "Irrigation required within 24 hrs (soil moisture low)."
    return "No irrigation recommended now."

def pest_advice(pest_probs):
    sorted_p = sorted(pest_probs.items(), key=lambda x: -x[1])
    top, prob = sorted_p[0]
    prob = float(prob)
    if prob < 0.4:
        return {"risk":"Low","message":f"Low risk of {top} ({prob:.2f}). Monitor."}
    elif prob < 0.75:
        return {"risk":"Medium","message":f"Moderate risk of {top} ({prob:.2f}). Scout and take preventive measures."}
    else:
        return {"risk":"High","message":f"High risk of {top} ({prob:.2f}). Consider control measures."}

def disease_advice(disease_probs, weather_summary=None):
    sorted_d = sorted(disease_probs.items(), key=lambda x: -x[1])
    top, prob = sorted_d[0]
    if prob > 0.6:
        return f"High risk of {top} ({prob:.2f}). Follow recommended fungicide and cultural controls."
    elif prob > 0.35:
        return f"Moderate risk of {top} ({prob:.2f}). Monitor closely."
    else:
        return f"Low disease risk ({prob:.2f})."

def generate_english_report(cell_id, location_name, prediction_dict):
    """
    prediction_dict contains keys like Crop_Health, Irrigation_Advice, Pest_Risk, Disease_Risk, Nutrient_Advice, AgroMet_Alert
    Saves a text report in outputs/reports/
    """
    lines = []
    lines.append(f"Advisory Report for {location_name or cell_id}")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
    lines.append("")
    lines.append(f"1. Crop status: {prediction_dict.get('Crop_Health')}")
    lines.append(f"2. Irrigation advice: {prediction_dict.get('Irrigation_Advice')}")
    pest = prediction_dict.get("Pest_Risk")
    if pest:
        lines.append(f"3. Pest advisory: {pest}")
    disease = prediction_dict.get("Disease_Risk")
    if disease:
        lines.append(f"4. Disease advisory: {disease}")
    nutr = prediction_dict.get("Nutrient_Advice")
    if nutr:
        lines.append(f"5. Nutrient advice: {nutr}")
    alert = prediction_dict.get("AgroMet_Alert")
    if alert:
        lines.append(f"6. AgroMet alert: {alert}")
    lines.append("")
    lines.append("Recommendation: Immediate monitoring is advised based on the above indicators.")
    text = "\n".join(lines)
    filename = f"{cell_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt"
    out_path = os.path.join(OUTPUT, "reports", filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, text
