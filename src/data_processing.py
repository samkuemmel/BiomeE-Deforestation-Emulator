import os
import numpy as np
import pandas as pd
import gcsfs

RESOLUTION = 1
N0_Lat, N0_Lon = 360, 720
N_Lat, N_Lon   = N0_Lat // RESOLUTION, N0_Lon // RESOLUTION
N_pfts, Npre, PI = 8, 6, 3.1415926
N_T_ANALYZE = 80 


PFT_ID   = ['C4G','C3G','TrE','TrD','TmE','TmD','Nfx','DeS']
PFTnames = ['C4 grass','C3 grass','Tropical evergreen','Tropical deciduous',
            'Temperate evergreen','Cold deciduous','Nitrogen fixers','Desert shrubs']
EcoVars = ['CAI','LAI','GPP','Rauto','Rh','Burned',
           'Tavg','Rain','SoilWater','Transp','Evap','Runoff',
           'plantC','soilC','plantN','soilN',
           'NSC','SeedC','leafC','rootC','swC','hwC',
           'NSN','SeedN','leafN','rootN','swN','hwN',
           'fineL','strucL','McrbC','fastSOC','slowSOC',
           'fineN','strucN','McrbN','fastSON','slowSON','mineralN',
           'WC1_5','WC2_25','WC3_50','WC4_100','WC5_120',
           'N_fxed','N_uptk','Nm_SL','Nm_FR','dNorg','dNgas','dNmin',
           'treeCA','grassCA','BMgrass','PET','Frisk','Pburn','CH4',
           'mu','muC','Indv']
N_gridV = len(EcoVars) - 3
EcoLongID = ['Crown area index','Leaf area index','Gross Primary Production',
             'Autotrophic respiration','Heterotrophic Respiration','Burned carbon',
             'Yearly mean temperature','Yearly rainfall','Soil water amount',
             'Yearly transpiration','Yearly evaporation','Yearly runoff',
             'Plant Biomass','Soil Organic Matter','Plant nitrogen','Soil nitrogen',
             'Non-structural carbon','Seed carbon','leaf carbon','root carbon',
             'Sapwood carbon','Heartwood carbon',
             'Non-structural nitrogen','Seed nitrogen','leaf nitrogen','root nitrogen',
             'Sapwood nitrogen','Heartwood nitrogen',
             'fineL','strucL','McrbC','fastSOC','slowSOC',
             'fineN','strucN','McrbN','fastSON','slowSON','mineralN',
             'WC1_5','WC2_25','WC3_50','WC4_100','WC5_120',
             'N_fixed','N uptake','Nm_SL','Nm_FR','Nloss1','Nloss2','Nloss3',
             'Woody crown area','Grass crown area','BMgrass',
             'Potential evapotranspiration','Fire risk','Fire probability','Methane',
             'Mortality rate','Mortality carbon flux','Woody individuals']
EcoUnit = ['m2/m2','m2/m2','KgC m-2 yr-1','KgC m-2 yr-1','KgC m-2 yr-1','KgC m-2 yr-1',
           'degree C','mm/year','mm','mm/year','mm/year','mm/year',
           'KgC m-2','KgC m-2','gN m-2','gN m-2',
           'KgC m-2','KgC m-2','KgC m-2','KgC m-2','KgC m-2','KgC m-2',
           'gN m-2','gN m-2','gN m-2','gN m-2','gN m-2','gN m-2',
           'KgC m-2','KgC m-2','KgC m-2','KgC m-2','KgC m-2',
           'gN m-2','gN m-2','gN m-2','gN m-2','gN m-2','gN m-2',
           'mm','mm','mm','mm','mm',
           'gN m-2 yr-1','gN m-2 yr-1','gN m-2 yr-1','gN m-2 yr-1','gN m-2 yr-1','gN m-2 yr-1','gN m-2 yr-1',
           'm2/m2','m2/m2','KgC m-2','mm/year','times/yr','times/yr','KgC m-2 yr-1',
           'fraction yr-1','KgC m-2 yr-1','individuals/m2']

_FS = None
def _fs():                                
    global _FS
    if _FS is None:
        _FS = gcsfs.GCSFileSystem()
    return _FS

def _gid(fname):                          
    return int(os.path.basename(fname)[Npre:Npre+6])

def _read_csv_gz(fname):
    with _fs().open(fname, 'rb') as raw:
        df = pd.read_csv(raw, compression='gzip', header=None, skiprows=1,
                         dtype=str, engine='c')
    return df.to_numpy(dtype=np.float64)

def process_eco(fname):
    """One ecosystem file -> (gid, year-series matrix of shape (N_T_ANALYZE, N_gridV))."""
    try:
        arr = _read_csv_gz(fname)         
        totYrs = arr.shape[0]
        if totYrs < N_T_ANALYZE:                    
            return (_gid(fname), None)
        return (_gid(fname), arr[-N_T_ANALYZE:, 2:2+N_gridV])
    except Exception:
        return (_gid(fname), None)        

def process_coh(fname):
    """One cohort file -> (gid, dict of per-PFT year-series matrices)."""
    try:
        CCYr = _read_csv_gz(fname)        
    except Exception:                     
        return (_gid(fname), None)        
    totCCL = CCYr.shape[0]
    if totCCL < 3:
        return (_gid(fname), None)
    totYrs = int(CCYr[:, 1].max())
    if totYrs < N_T_ANALYZE:
        return (_gid(fname), None)
        
    z = lambda: np.zeros((totYrs, N_pfts))
    GPP, NPP, BA, CA = z(), z(), z(), z()
    LA, BM, HT, den  = z(), z(), z(), z()
    mu, muC          = z(), z()
    for i in range(totCCL - 1):
        iYr = int(CCYr[i, 1]) - 1; iPFT = int(CCYr[i, 4]); iLayer = min(2, int(CCYr[i, 5]) - 1)
        GPP[iYr, iPFT] += CCYr[i, 6] * CCYr[i, 22] / 10000     
        NPP[iYr, iPFT] += CCYr[i, 6] * CCYr[i, 23] / 10000     
        BA[iYr, iPFT]  += CCYr[i, 6] * PI * 0.25 * CCYr[i, 11] ** 2
        LA[iYr, iPFT]  += CCYr[i, 6] * CCYr[i, 14] / 10000     
        BM[iYr, iPFT]  += CCYr[i, 6] * np.sum(CCYr[i, 15:21]) / 10000
        HT[iYr, iPFT]   = max(HT[iYr, iPFT], CCYr[i, 12])
        if iLayer == 0:                                        
            den[iYr, iPFT] += CCYr[i, 6]
            mu[iYr, iPFT]  += CCYr[i, 6] * CCYr[i, 29]
            muC[iYr, iPFT] += CCYr[i, 6] * CCYr[i, 29] * np.sum(CCYr[i, 15:21]) / 10000
            CA[iYr, iPFT]  += CCYr[i, 6] * CCYr[i, 13] / 10000
        if (iYr < int(CCYr[i + 1, 1]) - 1) or (i == totCCL - 2):   
            for j in range(N_pfts):
                mu[iYr, j] = mu[iYr, j] / den[iYr, j] if den[iYr, j] > 1e-4 else 0.0
                
    sl = slice(totYrs - N_T_ANALYZE, totYrs)
    return (_gid(fname), dict(
        GPP=GPP[sl], NPP=NPP[sl], BA=BA[sl], CA=CA[sl], LA=LA[sl], BM=BM[sl],
        HT=HT[sl], den=den[sl], mu=mu[sl], muC=muC[sl]))

def place(gid):                                           
    return (gid % 1000 - 1) // RESOLUTION, (gid // 1000 - 1) // RESOLUTION

def preprocess_modis(filepath):
    """Applies bitwise quality control and cloud masks to MODIS LAI/FPAR data."""
    import xarray as xr
    import numpy as np

    ds = xr.open_dataset(filepath, chunks={"time": "auto"})
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