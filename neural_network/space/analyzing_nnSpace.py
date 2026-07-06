import os
import json
from scipy.interpolate import griddata
from shapely.geometry import Point
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

# mapping packages
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import shape

# choose which biome to analyze
#selected_biome = "Amazon"
selected_biome = "Cerrado"

# force every simulation to use 1% annual deforestation
deforate = 1.0

# length of every autoregressive simulation
howmanyyears = 15

# location to save outputs
output_dir = os.path.expanduser("~/workflowsREU/spatialmaps/")
os.makedirs(output_dir, exist_ok=True)

##loading neural network 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNN(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)


# load the trained model and preprocessing scalers
# this is identical to the runspatial.py visualizer application so the
# emulator behaves exactly the same

def load_resources(biome):

    if biome == "Cerrado":

        scaler_X_path = os.path.expanduser("~/workflowsREU/models/CerradoscalerSpace_X.joblib")
        scaler_Y_path = os.path.expanduser("~/workflowsREU/models/CerradoscalerSpace_Y.joblib")
        modelpath = os.path.expanduser("~/workflowsREU/models/CerradoSpace_nnmodel")

    else:

        scaler_X_path = os.path.expanduser("~/workflowsREU/models/AmazonscalerSpace_X.joblib")
        scaler_Y_path = os.path.expanduser("~/workflowsREU/models/AmazonscalerSpace_Y.joblib")
        modelpath = os.path.expanduser("~/workflowsREU/models/AmazonSpace_nnmodel")

    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)

    model = SimpleNN(in_features=scaler_X.n_features_in_,out_features=scaler_Y.n_features_in_).to(device)

    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()

    return scaler_X, scaler_Y, model


# load the historical training data

def load_historical_data(biome):

    if biome == "Cerrado":
        path = os.path.expanduser("~/amazon_work/crmaster_training_datanoprcp.csv")
    else:
        path = os.path.expanduser("~/amazon_work/ammaster_training_datanoprcp.csv")

    return pd.read_csv(path)


scaler_X, scaler_Y, model = load_resources(selected_biome)
df_historical = load_historical_data(selected_biome)


weather_cols = ["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND"]
forceinput = ["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND","deforestation_pct", "lat", "lon"]

biomeinput = ["GPP", "LAI", "treeCA", "plantC",
              "soilC", "rootC", "mu", "Transp", "Indv"]

#physical bounds for biome variables, can tinker with it if you want
MAX_PHYSICAL_BOUNDS = np.array([
    20.0,
    15.0,
    5.0,
    100.0,
    200.0,
    10.0,
    1.0,
    5000.0,
    20000.0
])

boundary_path = os.path.expanduser(
    f"~/workflowsREU/{selected_biome.lower()}_boundary.geojson"
)

with open(boundary_path) as f:
    boundary_geojson = json.load(f)

#simulating at every lat lon pair

unique_locations = (
    df_historical[["lat", "lon"]]
    .drop_duplicates()
    .sort_values(["lat", "lon"])
    .reset_index(drop=True)
)

print(f"found {len(unique_locations)} unique grid cells")
results = []

print("beginning spatial simulations...")

with torch.no_grad():

    # loop over every unique location in the biome
    for i, location in unique_locations.iterrows():

        cell_lat = location["lat"]
        cell_lon = location["lon"]

        # grab the historical weather record for this pixel
        # this becomes the repeating climate cycle
        available_rates = df_historical["deforestation_pct"].unique()

        closest_defo = available_rates[
            np.abs(available_rates - deforate).argmin()
        ]
        
        df_cell = df_historical[
            (df_historical["lat"] == cell_lat) &
            (df_historical["lon"] == cell_lon) &
            (df_historical["deforestation_pct"] == closest_defo)
        ].sort_values("time")
        
        if len(df_cell) == 0:
            continue
        initial_row = df_cell.iloc[0]
        current_biome_state = initial_row[biomeinput].values.astype(float)
        starting_gpp = current_biome_state[0]
        num_records = len(df_cell)

        gpp_history = [starting_gpp]

        for year in range(howmanyyears):

            # cycle repeatedly through the available weather record

            row_this_year = df_cell.iloc[year % num_records].copy()

            weather_row = row_this_year[weather_cols].copy()

            # build the neural network input vector

            input_t = np.hstack([
                weather_row.values,
                [deforate],
                [cell_lat],
                [cell_lon],
                current_biome_state
            ])

            input_t = np.asarray(input_t, dtype=float)

            # skip impossible values before scaling

            if not np.all(np.isfinite(input_t)):
                print(f"invalid input at {cell_lat}, {cell_lon}")
                break

            # normalize inputs cause we're scaling them
            input_scaled = scaler_X.transform(input_t.reshape(1, -1))

            input_tensor = torch.tensor(
                input_scaled,
                dtype=torch.float32
            ).to(device)

            # neural network prediction

            pred_scaled = model(input_tensor)

            # transform back into physical units

            log_pred = scaler_Y.inverse_transform(
                pred_scaled.cpu().numpy()
            )[0]

            next_biome_state = np.expm1(log_pred)

            # keep predictions inside reasonable physical limits

            next_biome_state = np.clip(
                next_biome_state,
                a_min=0.0,
                a_max=MAX_PHYSICAL_BOUNDS
            )

            # advance the ecosystem forward one year
            current_biome_state = next_biome_state.copy()
            # save gpp for diagnostics
            gpp_history.append(current_biome_state[0])

   
        ending_gpp = current_biome_state[0]

        if starting_gpp > 0:

            percent_change = (
                (ending_gpp - starting_gpp)
                / starting_gpp
            ) * 100.0

        else:

            percent_change = np.nan

        # store one summary row for this location

        results.append({

            "lat": cell_lat,
            "lon": cell_lon,

            "gpp_start": starting_gpp,
            "gpp_end": ending_gpp,

            "gpp_change_pct": percent_change,
            "gpp_change_abs": ending_gpp - starting_gpp,

            "mean_gpp": np.mean(gpp_history),
            "min_gpp": np.min(gpp_history),
            "max_gpp": np.max(gpp_history)
        })
# convert into a dataframe

df_results = pd.DataFrame(results)


##save results 
output_csv = os.path.join(
    output_dir,
    f"{selected_biome.lower()}_gpp_spatial_response.csv"
)

df_results.to_csv(output_csv, index=False)

print(f"saved results to {output_csv}")