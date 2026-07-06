import os
import gzip
import shutil
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import torch
import torch.nn as nn
import joblib
from shapely.geometry import Point
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler         
from sklearn.model_selection import train_test_split 


##IMPORTANT: will build a clipped file later, change this between biome 
## you'll need to input what scenarios you want to do as well on line 131
#can update some path names if you're doing it for amazon or cerrado
biomeshapepath = os.path.expanduser("~/workflowsREU/cerrado_layer.shp")
#paths
BUCKET_BASE = "gs://leap-persistent/samkuemmel/Model_data/cru_subset/amazon/blk_0"
LOCAL_TMP = os.path.abspath("./amazon_work")
os.makedirs(LOCAL_TMP, exist_ok=True)

RAW_VARS = ["tmp", "dswrf", "pres", "pre", "spfh", "ugrd", "vgrd"]
biomeinput = ["GPP", "LAI", "treeCA", "plantC", "soilC", "rootC", "mu", "Transp", "Indv"]
## this is coming from BiomeE model time series integration steps, this has been clipped to the biome 
BIOMEE_OUTPUT_NC = os.path.expanduser("~/workflowsREU/cerrado002_cmp.nc")

YEARS = range(1991, 2021)  # 1991 to 2020 is the cru data range

def make_grid_points(lon_min, lat_min, lon_max, lat_max, step=0.5):
    lats = np.arange(lat_min, lat_max + step, step)
    lons = np.arange(lon_min, lon_max + step, step)
    return [(round(lat, 2), round(lon, 2)) for lat in lats for lon in lons]
# i pulled this from my legal biome clip geopandas boundaries
AMZN_BOUNDS = [-73.98318216, -16.6620185, -43.39931793, 5.26958083]
CERR_BOUNDS = [-60.47259563, -24.68178013, -41.27753553, -2.33208833]
GRID_POINTS = make_grid_points(*AMZN_BOUNDS) + make_grid_points(*CERR_BOUNDS)
print(len(GRID_POINTS), "grid points")

lats_arr = xr.DataArray([p[0] for p in GRID_POINTS], dims="points")
lons_arr = xr.DataArray([p[1] for p in GRID_POINTS], dims="points")


# forcing by taking all yearly and extracting it
def download_and_unzip(native_var, year):
    gz_name = f"crujra.v2.4.5d.{native_var}.{year}.365d.noc.nc.gz"
    gcs_path = f"{BUCKET_BASE}/{native_var}/{gz_name}"
    local_gz = os.path.join(LOCAL_TMP, gz_name)
    local_nc = local_gz[:-3]
    result = subprocess.run(["gsutil", "-q", "cp", gcs_path, local_gz], capture_output=True)
    if result.returncode != 0:
        print(f"  WARNING: missing {gcs_path}")
        return None
    with gzip.open(local_gz, "rb") as f_in, open(local_nc, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(local_gz)
    return local_nc

def get_all_points_series(local_nc):
    """Returns array shape (n_points, n_timesteps) for all grid points at once."""
    ds = xr.open_dataset(local_nc, engine="netcdf4")
    varname = list(ds.data_vars)[0]
    pts = ds[varname].sel(lat=lats_arr, lon=lons_arr, method="nearest")  # dims: (time, points)
    arr = pts.values.T  # -> (points, time)
    ds.close()
    return arr

# raw_yearly[var][year] = array (n_points, n_timesteps)
raw_yearly = {v: {} for v in RAW_VARS}
for native_var in RAW_VARS:
    for year in YEARS:
        local_nc = download_and_unzip(native_var, year)
        if local_nc is None:
            continue
        raw_yearly[native_var][year] = get_all_points_series(local_nc)
        os.remove(local_nc)
        print(f"{native_var} {year}: done")

# aggregate to yearly per point, derive RH/WIND
forcing_rows = []
for year in YEARS:
    if year not in raw_yearly["tmp"]:
        continue
    tmp_k = raw_yearly["tmp"][year]
    pres_pa = raw_yearly["pres"][year]
    spfh = raw_yearly["spfh"][year]
    u, v = raw_yearly["ugrd"][year], raw_yearly["vgrd"][year]
    wind = np.sqrt(u**2 + v**2)
    tmp_c = tmp_k - 273.15
    es = 611.2 * np.exp(17.67 * tmp_c / (tmp_c + 243.5))
    e = spfh * pres_pa / (0.622 + 0.378 * spfh)
    rh = np.clip(100 * e / es, 0, 100)

    for i, (lat, lon) in enumerate(GRID_POINTS):
        forcing_rows.append({
            "time": year, 
            "points": i,  # <--- Crucial: records the exact index position
            "lat": lat, 
            "lon": lon,
            "TEMP": np.mean(tmp_k[i]), 
            "Swdown": np.mean(raw_yearly["dswrf"][year][i]),
            "PRESSURE": np.mean(pres_pa[i]), 
            "RAIN": np.sum(raw_yearly["pre"][year][i]),
            "RH": np.mean(rh[i]), 
            "WIND": np.mean(wind[i]),
        })
df_forcing_all = pd.DataFrame(forcing_rows)
df_forcing_all.to_csv(os.path.join(LOCAL_TMP, "shared_forcing_1991_2020.csv"), index=False) # <-- Saved here!.          

#  Read your pre-computed weather baseline
df_forcing_all = pd.read_csv(os.path.join(LOCAL_TMP, "shared_forcing_1991_2020.csv"))


biome_shape = gpd.read_file(biomeshapepath).to_crs("EPSG:4326")

geometry = [Point(xy) for xy in zip(df_forcing_all["lon"], df_forcing_all["lat"])]
gdf_forcing = gpd.GeoDataFrame(df_forcing_all, geometry=geometry, crs="EPSG:4326")
gdf_forcing_clipped = gpd.sjoin(gdf_forcing, biome_shape, how="inner", predicate="within")

df_forcing_clean = pd.DataFrame(gdf_forcing_clipped).drop(
    columns=["geometry", "index_right"], errors="ignore")

unique_space = df_forcing_clean[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
lats_clean_arr = xr.DataArray(unique_space["lat"], dims="points")
lons_clean_arr = xr.DataArray(unique_space["lon"], dims="points")

#  Map out your deforestation files
# the number on the left must be a percentage out of 100
#eg 1.1% , you'd write 1.1
scenarios = {
    0.0: "~/workflowsREU/cerrado000_cmp.nc",
    0.2: "~/workflowsREU/cerrado002_cmp.nc",
    0.4: "~/workflowsREU/cerrado004_cmp.nc",
    0.55: "~/workflowsREU/cerrado0055_cmp.nc",
    0.6: "~/workflowsREU/cerrado006_cmp.nc",
    0.8: "~/workflowsREU/cerrado008_cmp.nc",
    #0.95: "~/workflowsREU/cerrado0095_cmp.nc",
    1.15: "~/workflowsREU/cerrado0115_cmp.nc",
    1.35: "~/workflowsREU/cerrado0135_cmp.nc",
    1.55: "~/workflowsREU/cerrado0155_cmp.nc",
    1.8: "~/workflowsREU/cerrado018_cmp.nc",
    2.2: "~/workflowsREU/cerrado022_cmp.nc",
    2.4: "~/workflowsREU/cerrado024_cmp.nc",
    2.55: "~/workflowsREU/cerrado0255_cmp.nc",
    #2.7: "~/workflowsREU/cerrado027_cmp.nc",
    2.9: "~/workflowsREU/cerrado029_cmp.nc",
    2.0: "~/workflowsREU/cerrado02_cmp.nc",
    3.1: "~/workflowsREU/cerrado031_cmp.nc",
    3.3: "~/workflowsREU/cerrado033_cmp.nc",
    3.5: "~/workflowsREU/cerrado035_cmp.nc",
    3.65: "~/workflowsREU/cerrado0365_cmp.nc",
    3.85: "~/workflowsREU/cerrado0385_cmp.nc",
    4.15: "~/workflowsREU/cerrado0415_cmp.nc",
    4.3: "~/workflowsREU/cerrado043_cmp.nc",
    4.45: "~/workflowsREU/cerrado0445_cmp.nc",
    4.6: "~/workflowsREU/cerrado046_cmp.nc",
    4.75: "~/workflowsREU/cerrado0475_cmp.nc",
    4.0: "~/workflowsREU/cerrado04_cmp.nc",
    5.0: "~/workflowsREU/cerrado05_cmp.nc",
}



all_scenario_dfs = []

# Prepare the forcing data frame with a matching alignment column
df_forcing_clean["cycle_index"] = df_forcing_clean["time"] - df_forcing_clean["time"].min()
for pct, nc_path in scenarios.items():
    print(f"Blending scenario: {pct}% deforestation...")
    
    ds_out = xr.open_dataset(os.path.expanduser(nc_path))
    out_pts = ds_out[biomeinput].sel(lat=lats_clean_arr, lon=lons_clean_arr, method="nearest")
    df_output_all = out_pts.to_dataframe().reset_index()
    #round to match clip sig figs
    if pd.api.types.is_datetime64_any_dtype(df_output_all['time']):
        df_output_all['time'] = df_output_all['time'].dt.year
        
    df_output_all = df_output_all.merge(
        unique_space, left_on="points", right_index=True
    )
    df_output_all = df_output_all.drop(columns=["points", "lat_x", "lon_x"], errors="ignore")
    df_output_all = df_output_all.rename(columns={"lat_y": "lat", "lon_y": "lon"})
    
    #TIME ALIGN
    # Shift forward simulation years (2021+) backward by 30 years to match the 1991-2020 weather forcing
    simulation_index = (df_output_all["time"] - df_output_all["time"].min()).astype(int)

    df_output_all["cycle_index"] = simulation_index % 30    
    # Merge using index values and matching weather timeline
    scenario_df = pd.merge(
        df_forcing_clean, 
        df_output_all, 
        on=["cycle_index", "lat", "lon"], 
        how="inner",
        suffixes=("_forcing", "_sim")
    )
    
    # Retain the accurate forward simulation timeline for tracking
    scenario_df["time"] = scenario_df["time_sim"]
    scenario_df["deforestation_pct"] = pct
    
    # Clean up the temporary alignment indicators and duplicated columns
    scenario_df = scenario_df.drop(columns=[ "time_forcing", "time_sim"], errors="ignore")
    scenario_df = scenario_df.dropna()
    
    all_scenario_dfs.append(scenario_df)

#Stack together and drop the temporary tracking index
master_training_df = pd.concat(all_scenario_dfs, ignore_index=True)
master_training_df = master_training_df.drop(columns=["points", "points_x", "points_y"], errors="ignore")

#Save file for the next cell
# this will be put into the master_df file path there
master_training_df.to_csv(os.path.join(LOCAL_TMP, "crmaster_training_data.csv"), index=False)
print("Master dataset compilation complete. Shape:", master_training_df.shape)