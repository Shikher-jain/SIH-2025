import pandas as pd

# Load CSVs
csv1 = pd.read_csv("ICRISAT-District Level Data.csv")
csv2 = pd.read_csv("Atal Jal 31 March 2021 .xlsx - Sheet1.csv")

# Optional: rename columns to match
csv1 = csv1.rename(columns={'State':'State Name','District':'Dist Name'})

# Merge on State and District
merged_df = pd.merge(csv2, csv1[['State Name','Dist Name','Latitude','Longitude']], 
                     on=['State Name','Dist Name'], 
                     how='left')  # use 'inner' if you want only matching districts

# Save merged CSV
merged_df.to_csv("merged_data.csv", index=False)
