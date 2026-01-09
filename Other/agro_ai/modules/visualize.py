# modules/visualize.py
import matplotlib.pyplot as plt
import os
import folium

OUTPUT = os.path.join(os.getcwd(), "outputs")
os.makedirs(os.path.join(OUTPUT, "graphs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "maps"), exist_ok=True)

def plot_indices_time_series(times, series_dict, name="indices"):
    out_path = os.path.join(OUTPUT, "graphs", f"{name}.png")
    plt.figure(figsize=(10,5))
    for k,v in series_dict.items():
        plt.plot(times, v, label=k)
    plt.xlabel("Time")
    plt.ylabel("Index value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def create_interactive_map(cell_points, labels, map_name="field_labels_map.html", zoom_start=12):
    """
    cell_points: list of dicts {"id":..., "lat":..., "lon":..., "popup": "..."}
    """
    if not cell_points:
        return None
    # center
    avg_lat = sum([p["lat"] for p in cell_points]) / len(cell_points)
    avg_lon = sum([p["lon"] for p in cell_points]) / len(cell_points)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_start, tiles="OpenStreetMap")
    color_map = {"Healthy":"green","Moderate":"orange","Stressed":"red"}
    for p in cell_points:
        color = color_map.get(p.get("label",""), "blue")
        folium.CircleMarker(location=[p["lat"], p["lon"]],
                            radius=8,
                            color="black",
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.8,
                            popup=folium.Popup(p.get("popup", p["id"]), parse_html=True)).add_to(m)
    out_path = os.path.join(OUTPUT, "maps", map_name)
    m.save(out_path)
    return out_path
