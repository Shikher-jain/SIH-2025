import os

def rename_files_by_split(folder):
    os.chdir(folder)
    
    for filename in os.listdir():
        if not filename.endswith(".npy"):
            continue

        parts = filename.split("_")
        
        if len(parts) < 7:
            print(f"❌ Skipping (not enough parts): {filename}")
            continue

        # Extract district, code, year
        district, code, year = parts[2:5]
        
        # Detect suffix (e.g., 'ndvi_heatmap.npy')
        suffix = "_".join(parts[5:])  # join the rest
        
        # New filename
        new_filename = f"{district}_{code}_{year}_{suffix}"
        
        os.rename(filename, new_filename)
        print(f"✅ Renamed: {filename} → {new_filename}")

# Use the function on your folder
folder_path = r"C:\shikher_jain\SIH\yeildModel\data\masknpy"
rename_files_by_split(folder_path)
