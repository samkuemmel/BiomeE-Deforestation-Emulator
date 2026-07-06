## Tools

Python • PyTorch • xarray • pandas • NumPy • Streamlit • NetCDF • NASA-GISS BiomeE • MODIS • PRODES

# Tropical Deforestation Analysis and Neural Network Emulator for the NASA-GISS BiomeE Model

## Final Deliverable:
Interactive streamlit visualizer allowing prediction at various locations, timelines, and deforestation rates in each biome
![Picture of interactive spatial dashboard](Visualizer/Visualizer_preview.png)


While the Amazon is well known, the Cerrado is Brazil's second largest biome and chronically understudied. This project compares impacts of deforestation between the two biomes using the NASA-GISS BiomeE model. Because a major limitation of the model is compute time, a neural network emulator is built on a new deforestation subroutine involving an annual hazard rate for each biome. This repository has the complete workflow, from finding hazard rates, to processing results, analyzing outputs, and training a neural network. Explanations for each notebook are also below.

## Workflow
## Basic Setup

This toolkit works directly with BiomeE's native configuration framework. Note: Hardcoded paths must be updated to the user, and will depend on how you implement the BiomeE run. Each .py will guide you through inputs, and often you will only need to change the username. 

1. **Clone & Build BiomeE:** Clone the BiomeE vegetation model (Weng et.al ) repository and follow its native instructions to set up the environment
2. **Apply Deforestation Update:** Replace the BiomeE harvest subroutine (around line 1830 of `src/vegetation.F90`) with the custom subroutine for deforestation pressure in `model_patch/HarvestUpdate.txt`.
3. **Forcing Data:** Download the crujra.v2.4.5d climate forcing data from [NCAR/UCAR](https://data.ucar.edu/) and update your local BiomeE configurations to point to these files. I used 1991-2020 data. 
4. **Analysis Environment:** Create the Python environment for this analysis toolkit using the provided `environment.yml`:
## Workflow
Part 1:
- Find historical deforestation rates using PRODES data
- Perform model runs at varying deforestation and precipitation (Sc_prcp in BiomeE) rates 
- Verify model outputs against MODIS data sourced from NASA AppEEars. To perform this step, you will need to request clipped-to-biome MCD15A2H.061 data (LAI)
- Evaluate historical scenarios in Pandas and Xarray, run collapse analysis for tipping points

Part 2: 
- Perform forcing and compression for the by-biome (Flat) neural network, then run network
- Analyze neural network outputs to better find tipping points, run SHAP interpolation
- Run spatiotemporal (Space) forcing scripts, neural network, and analysis
- Run respective neural networks in Streamlit using Visualizer scripts


## Repository Structure


| File | Description |
|------|-------------|
| `model_patch/HarvestUpdate.txt` | Replace the deforestation subroutine of the BiomeE NASA vegetation model (around line 1830 of `src/vegetation.F90` of his model) |
| `config/environment.yml` | Loads necessary modules
| `src/data_processing.py` | Assisting compiling data from an output bucket into a NetCDF|
| `workflows/deforestation_rates.py` | Finds deforestation rates across Brazil to identify variance across policy eras |
| `workflows/netcdf_compile_workflow.py` | Integrates results into a NetCDF |
| `workflows/biome_clip.py` | Clips NetCDF results to legal biome boundaries |
| `workflows/MODIS_work.py` | Compares BiomeE results to NASA satellite data |
| `workflows/biome_comparison_analysis.py` | Plots and runs breakpoint and AC-1 analysis on BiomeE results |
| `workflows/explore_one_grid.py` | Plots and analyzes a single grid cell as a sanity check |
| `neural_network/forcing/nn_forcing.py` | Prepares inputs for neural network. |
| `neural_network/forcing/nn_forcing_precipitation.py` | Prepares inputs for neural network, with precipitation, used when running precipitation scaling |

| `neural_network/flat/neural_networkFlat.py` |  trains by-biome PyTorch neural network, saves weights. Incorporates precipitation by default.  |
| `neural_network/space/neural_networkSpace.py` |  trains spatio-temporal PyTorch neural network, saves weights. Incorporates precipitation by default. |
| `neural_network/space/neural_networkSpace-noprecip.py` | trains spatio-temporal PyTorch neural network, saves weights. Does not incorporate precipitation for simpler training | 

| `neural_network/flat/analyzing_nnFlat.py` | Plots and analyzes neural network predictions by-biome, runs SHAP values, all loading from saved weights |
| `neural_network/space/analyzing_nnSpace.py` | Runs, saves simulations for each lat/lon pair to analyze as desired |
| `neural_network/flat/compress_neural_forcing.py` | Compresses neural network forcing data, removing the spatial component|
| `visualizer/appflat.py` | Visualizes by-biome neural network. Can customize with images of your choice |
| `visualizer/appspatial.py` | Visualizes spatiotemporal neural network. Can customize with images of your choice |