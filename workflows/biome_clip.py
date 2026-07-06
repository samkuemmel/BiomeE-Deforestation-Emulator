#This code clips time series integrate to biomes
#uncomment whichever biome you want to do
import xarray as xr


sf1 = '/home/jovyan/output/BiomeE_Simu_Cerrado031.nc'


ds = xr.open_dataset(sf1)

ds = ds.assign_coords(
    lon=ds["Longitude"].values,
    lat=ds["Latitude"].values
)

import geopandas as gpd
import regionmask

#amazon = gpd.read_file("~/workflowsREU/amazon_layer.shp").to_crs("EPSG:4326")
cerrado = gpd.read_file("~/workflowsREU/cerrado_layer.shp").to_crs("EPSG:4326")


#amask = regionmask.mask_geopandas(
    #amazon,
    #ds.lon,
    #ds.lat
#)

#amazon_ds = ds.where(~amask.isnull())

cmask = regionmask.mask_geopandas(
    cerrado,
    ds.lon,
    ds.lat
)

cerrado_ds = ds.where(~cmask.isnull())
#amazon_ds.to_netcdf("amazon009.nc")
cerrado_ds.to_netcdf("cerrado031.nc")