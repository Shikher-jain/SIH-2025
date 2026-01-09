import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_cell_dashboard(cell_id, pred_label, indices_dict, weather_dict, report_text, out_dir="outputs/dashboards"):
    """
    Generates a dashboard figure per cell containing:
    - Crop label heatmap
    - Bar plot for all indices
    - Weather summary
    - Advisory report
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Map label to color/text
    label_map = {0: "Stressed", 1: "Moderate", 2: "Healthy"}
    label_color_map = {0: "red", 1: "orange", 2: "green"}
    label_text = label_map.get(pred_label, "Unknown")
    label_color = label_color_map.get(pred_label, "gray")

    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    
    # 1️⃣ Crop label
    axs[0,0].barh([0], [1], color=label_color)
    axs[0,0].set_xlim(0,1)
    axs[0,0].set_yticks([])
    axs[0,0].set_title(f"Predicted Crop Condition: {label_text}", fontsize=14)

    # 2️⃣ Indices bar plot
    indices_names = [k for k in indices_dict.keys() if "_mean" in k]
    indices_values = [indices_dict[k] for k in indices_names]
    sns.barplot(x=indices_values, y=indices_names, ax=axs[0,1], palette="viridis")
    axs[0,1].set_title("Vegetation Indices (Mean Values)")

    # 3️⃣ Weather summary
    weather_text = "\n".join([f"{k}: {v:.1f}" for k,v in weather_dict.items()])
    axs[1,0].text(0.1, 0.5, weather_text, fontsize=12)
    axs[1,0].axis("off")
    axs[1,0].set_title("Weather Summary", fontsize=14)

    # 4️⃣ Advisory report
    axs[1,1].text(0.05, 0.95, report_text, fontsize=11, va="top")
    axs[1,1].axis("off")
    axs[1,1].set_title("Advisory Report", fontsize=14)

    plt.suptitle(f"Dashboard for {cell_id}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    
    # Save figure
    out_path = os.path.join(out_dir, f"{cell_id}_dashboard.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
