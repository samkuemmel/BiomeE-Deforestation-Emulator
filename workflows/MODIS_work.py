## this script requires MODIS data from NASA AppEEARS, specifically MCD15A2H.061


import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*FrozenMappingWarningOnValuesAccess.*")

def preprocess_modis(filepath):
    ds = xr.open_dataset(filepath, chunks={"time":"auto"})
    qc = ds["FparLai_QC"].fillna(225).astype(np.uint8)
    extra_qc = ds["FparExtra_QC"].fillna(225).astype(np.uint8)
    
    bit0 = xr.apply_ufunc(np.bitwise_and, qc, 1, dask="allowed") == 0
    bits5_7 = (
        xr.apply_ufunc(np.bitwise_right_shift, qc, 5, dask="allowed")
    ) & 0b00000111
    fpar_lai_mask = (bit0 == True) & (bits5_7 == 0)

    bits3_4 = (
        xr.apply_ufunc(np.bitwise_right_shift, extra_qc, 3, dask="allowed")
    ) & 0b00000011
    cloud_mask = bits3_4 == 0

    good_mask = fpar_lai_mask & cloud_mask

    ds["Lai_500m"] = ds["Lai_500m"].where(good_mask)
    ds["Fpar_500m"] = ds["Fpar_500m"].where(good_mask)

    if ds["Lai_500m"].max(skipna=True) > 100:
        ds["Lai_500m"] = ds["Lai_500m"] * 0.1
        ds["Fpar_500m"] = ds["Fpar_500m"] * 0.01
    print("aggregating yrly")
    ds_yearly = ds[["Lai_500m", "Fpar_500m"]].resample(time="YS").mean()
    return ds_yearly.compute()

amod = preprocess_modis("~/Modis/AmazonModis.nc")
print("LAI Range:", float(amod["Lai_500m"].min()), "to", float(amod["Lai_500m"].max()))
print("FPAR Range:", float(amod["Fpar_500m"].min()), "to", float(amod["Fpar_500m"].max()))
amod.to_netcdf("~/Modis/AmazonModis_Clean_Yearly.nc")

ads = xr.open_dataset('/home/jovyan/workflowsREU/amazon002.nc')
cds = xr.open_dataset('/home/jovyan/Modis/AmazonModis_Clean_Yearly.nc')
print(list(cds.data_vars))

ads_mean = ads.mean(dim=("lat", "lon"))
cds_mean = cds[["Lai_500m"]].mean(dim=("lat", "lon"))  

ayr = ads_mean.to_dataframe().reset_index().rename(columns={"time": "yr"})
cyr = cds_mean.to_dataframe().reset_index().rename(columns={"time": "yr"})
ayr.columns = ayr.columns.str.strip()
cyr.columns = cyr.columns.str.strip()

ayr["NPP"] = ayr["GPP"] - ayr["Rauto"]
primary_vars = ["yr", "GPP", "NPP", "treeCA", "Indv", "mu", "LAI", "Transp", "plantC", "fineL", "rootC", "soilC"]
adata = ayr[primary_vars].copy().rename(columns={"treeCA": "Acrown", "Indv": "N_ha"})
cdata = cyr[["yr", "Lai_500m"]].copy().rename(columns={"Lai_500m": "LAI"})

vars_ads = ["GPP", "NPP", "Acrown", "N_ha", "mu", "LAI", "Transp", "plantC", "fineL", "rootC", "soilC"]
anewdf = adata.groupby("yr")[vars_ads].mean().reset_index()
cnewdf = cdata.groupby("yr")[["LAI"]].mean().reset_index() 

acurrentvar = "LAI"
ccurrentvar = "LAI"
acurrentvar2 = "LAI"
ccurrentvar2 = "LAI"
abreak = 0
cbreak = 0

plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
years = range(2021, 2026)
ROLL = 1

a_series2 = pd.Series(anewdf[acurrentvar2].values[1:6], index=years)
aroll2 = a_series2.rolling(ROLL, center=True, min_periods=1).mean()
adiff2 = -1 * (a_series2.iloc[0] - a_series2.iloc[-25:].mean()) / a_series2.iloc[0]
print(f"___ {acurrentvar2} difference: {adiff2}")
ax[0, 0].plot(years, a_series2, alpha=0.3, color="#4f772d")
ax[0, 0].plot(years, aroll2, color="#4f772d", linewidth=2, label=f"Cerrado 0.95% {acurrentvar2}")
if abreak > 0:
    ax[0, 0].axvline(x=abreak, color="red", linestyle="--", label=f"Breakpoint ({int(abreak)})")
ax[0, 0].set_title(f"___ {acurrentvar2} Over Time")

a_series = pd.Series(anewdf[acurrentvar].values[1:6], index=years)
aroll = a_series.rolling(ROLL, center=True, min_periods=1).mean()
adiff = -1 * (a_series.iloc[0] - a_series.iloc[-25:].mean()) / a_series.iloc[0].mean()
print(f"___ {acurrentvar} difference: {adiff}")
ax[0, 1].plot(years, a_series, alpha=0.3, color="#4f772d")
ax[0, 1].plot(years, aroll, color="#4f772d", linewidth=2, label=f"Cerrado 0.95% {acurrentvar}")
if abreak > 0:
    ax[0, 1].axvline(x=abreak, color="red", linestyle="--", label=f"Breakpoint ({int(abreak)})")
ax[0, 1].set_title(f"___ {acurrentvar} Over Time")

c_series2 = pd.Series(cnewdf[ccurrentvar2].values[1:6], index=years)
croll2 = c_series2.rolling(ROLL, center=True, min_periods=1).mean()
cdiff2 = -1 * (c_series2.iloc[0] - c_series2.iloc[-25:].mean()) / c_series2.iloc[0]
print(f"___ {ccurrentvar2} difference: {cdiff2}")
ax[1, 0].plot(years, c_series2, alpha=0.3, color="#4f772d")
ax[1, 0].plot(years, croll2, color="#4f772d", linewidth=2, label=f"Cerrado 0.95% {ccurrentvar2}")
if cbreak > 0:
    ax[1, 0].axvline(x=cbreak, color="red", linestyle="--", label=f"Breakpoint ({int(cbreak)})")
ax[1, 0].set_title(f"___ %  {ccurrentvar2} Over Time")

c_series = pd.Series(cnewdf[ccurrentvar].values[1:6], index=years)
croll = c_series.rolling(ROLL, center=True, min_periods=1).mean()
cdiff = -1 * (c_series.iloc[0] - c_series.iloc[-25:].mean()) / c_series.iloc[0]
print(f"___ {ccurrentvar} difference: {cdiff}")
ax[1, 1].plot(years, c_series, alpha=0.3, color="#4f772d")
ax[1, 1].plot(years, croll, color="#4f772d", linewidth=2, label=f"Cerrado 0.95% {ccurrentvar}")
if cbreak > 0:
    ax[1, 1].axvline(x=cbreak, color="red", linestyle="--", label=f"Breakpoint ({int(cbreak)})")
ax[1, 1].set_title(f"___ %  {ccurrentvar} Over Time")

plt.tight_layout()
plt.show()