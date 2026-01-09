# modules/report.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from modules.dataset_builder import OUTPUT_DIR

def generate_advisory_report(df):
    report_file = os.path.join(OUTPUT_DIR,"reports","advisory_report.txt")
    with open(report_file,"w") as f:
        f.write("Advisory Report:\n\n")
        for _, row in df.iterrows():
            f.write(f"{row['cell_id']}:\n")
            ndvi = row.get("NDVI_mean",0)
            if ndvi>0.6: f.write("1. Crop condition healthy.\n")
            elif ndvi>0.35: f.write("1. Crop condition shows moderate stress.\n")
            else: f.write("1. Crop condition shows high stress.\n")
            if row.get("precipitation",0)<25: f.write("2. Irrigation required soon.\n")
            if row.get("humidity",0)>75: f.write("3. Risk of fungal disease high.\n")
            f.write("\n")
    print(f"✅ Advisory report saved at {report_file}")
    return report_file

def plot_crop_label_heatmap(df):
    plt.figure(figsize=(8,6))
    pivot = df.pivot("lat","lon","label")
    sns.heatmap(pivot, cmap="RdYlGn", cbar_kws={'label':'Crop Label'})
    heatmap_file = os.path.join(OUTPUT_DIR,"reports","crop_label_heatmap.png")
    plt.title("Crop Condition Heatmap")
    plt.savefig(heatmap_file)
    plt.show()
    print(f"✅ Heatmap saved at {heatmap_file}")
    return heatmap_file
