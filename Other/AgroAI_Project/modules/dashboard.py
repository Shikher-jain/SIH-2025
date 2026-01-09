import matplotlib.pyplot as plt
import seaborn as sns
import os
def generate_cell_dashboard(cell_id, label):
    os.makedirs('outputs/dashboards', exist_ok=True)
    path = os.path.join('outputs','dashboards', f'{cell_id}_dashboard.png')
    plt.figure(figsize=(6,2))
    sns.heatmap([[label]], annot=True, cmap='RdYlGn', cbar=True)
    plt.title(f'Cell {cell_id} Condition (0=Stressed,1=Moderate,2=Healthy)')
    plt.savefig(path)
    plt.close()
    return path
