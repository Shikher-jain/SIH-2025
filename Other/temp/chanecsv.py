import pandas as pd

# Step 1: Load original CSV
input_path = "data/Target/Punjab&UP_Yield_2018To2021.csv"  # 👈 Replace with your actual file path
df = pd.read_csv(input_path)

# Step 2: Drop unwanted columns
columns_to_drop = ["State", "Area_Hectare", "Production_Tonnes"]
df_clean = df.drop(columns=columns_to_drop)

# Step 3: Save the cleaned data to a new CSV
output_path = "data/Target/Punjab&UP_Yield_2018To2021.csv"  # 👈 Change output path if needed
df_clean.to_csv(output_path, index=False)

# Optional: Show first few rows
print("Cleaned CSV saved successfully!")
print(df_clean.head())
