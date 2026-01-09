# modules/predict.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from modules.indices import compute_indices_from_image, resize_image
from modules.weather import parse_weather_json
from modules.advisory import irrigation_advice, pest_advice, disease_advice, generate_english_report
from modules.visualize import plot_indices_time_series, create_interactive_map

ROOT = os.path.abspath(os.getcwd())
OUTPUT = os.path.join(ROOT, "outputs")
os.makedirs(os.path.join(OUTPUT, "predictions"), exist_ok=True)

def load_rf_model():
    import joblib
    path = os.path.join(OUTPUT, "models", "rf_model.pkl")
    mapping_path = os.path.join(OUTPUT, "models", "label_mapping.pkl")
    rf = None
    mapping = None
    if os.path.exists(path):
        rf = joblib.load(path)
    if os.path.exists(mapping_path):
        mapping = joblib.load(mapping_path)
    return rf, mapping

def single_predict_from_paths(sat_tif, metadata_json=None, weather_json=None):
    import rasterio
    if not os.path.exists(sat_tif):
        raise FileNotFoundError(sat_tif)
    with rasterio.open(sat_tif) as src:
        arr = src.read()
        img = np.transpose(arr, (1,2,0))
    img_resized = resize_image(img, target=(64,64))
    indices = compute_indices_from_image(img)

    w = {}
    if weather_json:
        w = parse_weather_json(weather_json if isinstance(weather_json, dict) else json.load(open(weather_json)))
    else:
        w = parse_weather_json({})  # will create mock

    seq24 = w["seq24"]
    seq_arr = np.array([[s["temperature"], s["humidity"], s["precipitation"], s["windSpeed"], s["pressure"]] for s in seq24])

    rf, mapping = load_rf_model()
    # prepare tab features for RF
    tab = {k: indices.get(k, 0) for k in indices.keys()}
    # add current weather
    tab["temperature"] = w["current"]["temperature"]
    tab["humidity"] = w["current"]["humidity"]
    tab["precipitation"] = w["current"]["precipitation"]
    # convert to df with same order as RF expects (if available)
    if rf is None:
        rf_label = "Unknown(no RF)"
    else:
        # get feature names if present
        try:
            cols = rf.feature_names_in_
            X = pd.DataFrame([tab]).reindex(columns=cols, fill_value=0)
        except Exception:
            X = pd.DataFrame([tab]).fillna(0)
        code = int(rf.predict(X)[0])
        rf_label = mapping.get(code, str(code)) if mapping else str(code)

    # simple mock pest/disease heuristics (should be replaced with trained models)
    t = w["current"]["temperature"]
    h = w["current"]["humidity"]
    pest_probs = {"Aphids":0.1,"Whitefly":0.05,"FallArmyworm":0.02}
    if 20 <= t <= 32 and 40 <= h <= 80:
        pest_probs["Aphids"] = 0.65
    pest_info = pest_advice(pest_probs)

    disease_probs = {"Rice Blast":0.1,"Wheat Rust":0.05}
    if h > 85 and 18 < t < 28:
        disease_probs["Rice Blast"] = 0.76
    disease_msg = disease_advice(disease_probs)

    # irrigation advice
    soil_moisture = tab.get("soil_moisture", 30)
    forecast_precip_mm = sum([s["precipitation"] for s in seq24[:6]])
    leaf_status = "Wet" if (w["current"]["humidity"] > 90 or w["current"]["precipitation"] > 0) else ("Mild Wetness" if w["current"]["humidity"]>80 else "Dry")
    irrigation = irrigation_advice(soil_moisture, forecast_precip_mm, leaf_status)

    nutrient_advice = None
    if indices.get("NDVI_mean",0) < 0.35:
        nutrient_advice = "Apply Nitrogen fertilizer (Urea) per recommended dose."

    out = {
        "metadata": {
            "cell_id": os.path.basename(sat_tif).split('.')[0],
            "lat": (metadata_json.get("lat") if metadata_json else None),
            "lon": (metadata_json.get("lon") if metadata_json else None),
            "downloadTime": datetime.utcnow().isoformat() + "Z",
            "apiSource": "OpenWeather-like"
        },
        "Crop_Health": rf_label,
        "Irrigation_Advice": irrigation,
        "Pest_Risk": pest_info["message"],
        "Disease_Risk": disease_msg,
        "Nutrient_Advice": nutrient_advice,
        "AgroMet_Alert": f"Forecast next 6h precipitation: {forecast_precip_mm:.2f} mm"
    }

    # save JSON and produce English report
    pred_name = f"prediction_{os.path.basename(sat_tif).replace('.','_')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    json_path = os.path.join(OUTPUT, "predictions", pred_name + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # create English report text
    report_path, report_text = generate_english_report(out["metadata"]["cell_id"], metadata_json.get("name") if metadata_json else None, out)

    # create visuals
    times = [s["timestamp"] for s in seq24]
    series = {"NDVI":[indices.get("NDVI_mean",0)]*len(times),
              "SAVI":[indices.get("SAVI_mean",0)]*len(times),
              "EVI":[indices.get("EVI_mean",0)]*len(times)}
    graph_path = plot_indices_time_series(times, series, name=pred_name + "_indices")
    # create interactive map for a single cell
    map_path = None
    if metadata_json and "lat" in metadata_json and "lon" in metadata_json:
        point = [{"id": out["metadata"]["cell_id"], "lat": float(metadata_json["lat"]), "lon": float(metadata_json["lon"]), "popup": out["Crop_Health"], "label": out["Crop_Health"]}]
        map_path = create_interactive_map(point, labels=None, map_name=pred_name + "_map.html")

    # return output
    return {"json": json_path, "report": report_path, "graph": graph_path, "map": map_path, "prediction": out}

def interactive_predict():
    sat = input("Enter satellite tif path: ").strip()
    meta = input("Enter metadata json path (or Enter to skip): ").strip() or None
    weather = input("Enter weather json path (or Enter to skip): ").strip() or None
    meta_json = None
    if meta:
        import json
        meta_json = json.load(open(meta,'r',encoding='utf-8'))
    weather_obj = None
    if weather:
        weather_obj = json.load(open(weather,'r',encoding='utf-8'))
    result = single_predict_from_paths(sat, metadata_json=meta_json, weather_json=weather_obj)
    print("Prediction saved:", result)
