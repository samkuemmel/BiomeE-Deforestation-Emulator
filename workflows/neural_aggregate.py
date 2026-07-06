## Run this after creating the forcing data in nn_forcing. This will allow you to use the forcing from the spatiotemporal run and quickly integrate it into a spaceless model
##Aggregates spatially within each year 
## if you ran the forcing without precip_scale, make sure to comment that out
import os
import pandas as pd

# Define paths
lcl = os.path.expanduser("~/amazon_work")
input_path = os.path.join(lcl, "ammaster_training_data.csv")
output_path = os.path.join(lcl, "amazon_spatial_averaged.csv")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find the master dataset at {input_path}.")

print("Loading master dataset...")
df = pd.read_csv(input_path)


cols_to_drop = ["lat", "lon", "biome"] 

df_averaged = (df.groupby(["time", "deforestation_pct", "precip_scale"], as_index=False).mean(numeric_only=True)) ## remove precip scale if training without precipitation
#extra check
df_averaged = df_averaged.drop(columns=cols_to_drop, errors="ignore")
cols = ["cycle_index", "deforestation_pct", "precip_scale"] + [c for c in df_averaged.columns if c not in ["cycle_index", "deforestation_pct","precip_scale"]] # remove precip_scale if not training precipitation 
df_averaged = df_averaged[cols]

#save
df_averaged.to_csv(output_path, index=False)
print(f"Successfully saved spatially averaged dataset to: {output_path}")