import pandas as pd

# Load CSV (jisme columns: District, Year, Yield)
input_path = r"C:\shikher_jain\SIH\yeildModel\data\yield\Punjab&UP_Yield_2018To2021.csv"
df = pd.read_csv(input_path)

# Sort by District ascending, then Year ascending

df_sorted = df.sort_values(by=["District", "Year"], ascending=[True , True])

# Save sorted file
output_path = r"C:\shikher_jain\SIH\yeildModel\data\yield\Punjab&UP_Yield_2018To2021.csv"
df_sorted.to_csv(output_path, index=False)

print("CSV sorted by District and Year ascending saved successfully!")
print(df_sorted.head())