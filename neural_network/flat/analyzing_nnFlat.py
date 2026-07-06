#this saves the time of retraining a model once it has already been saved
## dont need if you've just ran the neural network steps from neural_network.py
import os
import warnings
import glob
import numpy as np
import torch
import pandas as pd
import ewstools
import pwlf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import torch.nn as nn
import joblib
import shap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "sans-serif"

# set data domain token to auto align saving paths
domain_short = "amazon" # change to cerrado when switching datasets #cr or am
domain_prefix = "Amazon" # Cerrado or Amazon
r_recycling = 0.28 if domain_prefix == "Amazon" else 0.15

df_historical = pd.read_csv(f"~/amazon_work/{domain_short}_spatial_averaged.csv")
# Build once near the top of the script

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)
scaler_X = os.path.expanduser(f"~/workflowsREU/models/{domain_prefix}scalerFlat_X.joblib")
scaler_Y = os.path.expanduser(f"~/workflowsREU/models/{domain_prefix}scalerFlat_Y.joblib")
scaler_X = joblib.load(scaler_X)
scaler_Y = joblib.load(scaler_Y)
print("Scalers successfully loaded!")
model = SimpleNN(in_features = scaler_X.n_features_in_, out_features = scaler_Y.n_features_in_).to(device)

# Load the saved state weights into the model variable
modelpath = os.path.expanduser(f"~/workflowsREU/models/{domain_prefix}Flat_nnmodel")
model.load_state_dict(torch.load(modelpath, map_location=device))
model.eval() # Set eval to use

print("Model successfully loaded into variable 'model'!")

forceinput = ["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND", "deforestation_pct"]
biomeinput = ["GPP", "LAI", "treeCA", "plantC", "soilC", "rootC", "mu", "Transp", "Indv"]

X_numpy = df_historical[forceinput + biomeinput].values
# Make sure you only select the exact input columns your model was trained on!
# 
print(f"X_numpy successfully recreated with shape: {X_numpy.shape}")##SHAP
#This explains how important each feature is to the emulator

#get an index to put into the explainer, and one for the values
idxfeed = np.random.choice(len(X_numpy), 100, replace=False)
idxvalues = np.random.choice(len(X_numpy), 200, replace=False)

#loads tensors, feeds it and plots it 
tensorexplain= torch.tensor(scaler_X.transform(X_numpy[idxfeed]),dtype = torch.float32).to(device)
tensortest= torch.tensor(scaler_X.transform(X_numpy[idxvalues]),dtype = torch.float32).to(device)
explainer = shap.GradientExplainer(model, tensorexplain)
shapvalues = explainer.shap_values(tensortest)

#explains how much a feature contributres to prediction error/movement
shapplot = shap.summary_plot(
    shapvalues, 
    tensortest.cpu().numpy(), 
    feature_names=forceinput + biomeinput,
    show=False
)

fig = plt.gcf()
fig.subplots_adjust(top=0.88)  # reserve space above the plot
fig.text(0.5, 0.97, f"{domain_prefix} SHAP Feature Importance", fontsize=16, ha="center", va="top", fontweight="bold")
plt.show()
# Make sure you only select the exact input columns your model was trained on!
# 
print(f"X_numpy successfully recreated with shape: {X_numpy.shape}")##SHAP
#This explains how important each feature is to the emulator

## this goes and plots it all 

biomevariables = biomeinput
shapdf = []

# pick one output (e.g. LAI = index 1)
shapv = shapvalues[1]

# total feature list alignment
combined_features = forceinput + biomeinput

for idx in range(len(biomevariables)):

    # get feature name
    var = biomevariables[idx]

    # pull SHAP values for that feature across all samples
    vals = shapv[:, idx]

    # stats
    meanv = vals.mean()
    stdv = vals.std()
    meanabs = np.abs(vals).mean()

    shapdf.append([var, meanv, stdv, meanabs])

    plt.figure()
    plt.hist(vals, bins=50)
    plt.xlabel("SHAP value")
    plt.ylabel("Count")
    plt.title("Distribution of SHAP values: " + var)
    plt.show()

shapcsv = pd.DataFrame(shapdf, columns=["feature", "mean_shap", "std_shap", "mean_abs_shap"])
## this dataframe has all the statistical values 
shapcsv.head()
#shapcsv.to_csv("shap_results.csv", index=False)


## LOOP TO RUN SPECIFIC DEFORESTATION PERCENTAGES**

# SET YOUR SIMULATION DIALS
howmanyyears = 50
biomeinput = ["GPP", "LAI", "treeCA", "plantC", "soilC", "rootC", "mu", "Transp", "Indv"]
vars_to_analyze = ["plantC", "soilC","GPP"]
#setting seed basically
sample_idx = 1

# dense sweep instead of one deforate at a time - this is the whole point
# of having an emulator, so don't be shy about how many points you throw at it
# Average climate forcing for each historical year
weather_by_year = (
    df_historical
    .groupby("time")[["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND"]]
    .mean()
    .sort_index()
)

T_baseline = df_historical["Transp"].mean()

# NEW: sweep over rainfall as well
defo_sweep = np.linspace(0.1, 3.0, 60)
rain_sweep = np.linspace(0.60, 1.20, 30)      # 60%–120% historical rainfall


ac1_records = []
model.eval()
for rain_multiplier in rain_sweep:

    print(f"\nRain multiplier = {rain_multiplier:.2f}")

    for deforate in defo_sweep:

        initial_row = X_numpy[sample_idx].copy()
        current_biome_state = initial_row[len(forceinput):].copy()

        transp_idx = biomeinput.index("Transp")
        prev_transp = T_baseline

        sim_results = []

        with torch.no_grad():

            for yr in range(howmanyyears):

               #climate force 
                weather_row = weather_by_year.iloc[
                    yr % len(weather_by_year)
                ].copy()

                #  rainfall chosen by sweep
                P_external = weather_row["RAIN"] * rain_multiplier

                # Moisture recycling feedback
                t_rel = np.clip(prev_transp / T_baseline, 0.0, 1.0)

                F_rain = 1.0 - r_recycling * (1.0 - t_rel)

                rain_dynamic = P_external * F_rain

                weather_row["RAIN"] = rain_dynamic

                # Build forcing vector
                current_force = weather_row.copy()
                current_force["deforestation_pct"] = deforate

                x_in = np.hstack([
                    current_force.values,
                    current_biome_state
                ])

                x_scaled = scaler_X.transform(
                    x_in.reshape(1, -1)
                )

                prediction = model(
                    torch.tensor(
                        x_scaled,
                        dtype=torch.float32,
                        device=device
                    )
                )

                current_biome_state = scaler_Y.inverse_transform(
                    prediction.cpu().numpy()
                )[0]

                prev_transp = current_biome_state[transp_idx]

                result = dict(zip(biomeinput, current_biome_state))

                result["Year"] = yr
                result["Deforestation"] = deforate
                result["RainMultiplier"] = rain_multiplier
                result["RAIN_external"] = P_external
                result["RAIN_coupled"] = rain_dynamic

                sim_results.append(result)
    ##CHANGE OUTPUT PATHS WHEN YOU DO Cerrado/Amazon switch!!!
            df_sim = pd.DataFrame(sim_results)
            output_path = os.path.expanduser(
                f"~/workflowsREU/nnsims/"
                f"{domain_prefix}simulation_"
                f"rain{rain_multiplier:.2f}_"
                f"defo{deforate:.3f}_"
                f"{howmanyyears}years.csv"
            )
            df_sim.to_csv(output_path, index=False)
            print(f"Saved simulation results to {output_path}")

    # now pull ac1 for this defo level, same detrend/rolling setup as before,
    # just collapsed down to one trend-strength number per variable
            for test_var in vars_to_analyze:
                df_clean = df_sim[["Year", test_var]].dropna().set_index("Year")
                series_test = df_clean[test_var]
                if len(series_test) <= 15:
                    continue
                try:
                    ts_test = ewstools.TimeSeries(series_test, transition=series_test.index[-15])
                    ts_test.detrend(method="Gaussian", bandwidth=0.2)
                    ts_test.compute_auto(lag=1, rolling_window=0.2)
                    ts_test.compute_var(rolling_window=0.2)
                    ts_test.compute_ktau()
                    ews = ts_test.ews.copy()
                    mean_ac1 = ews["ac1"].mean()
                    ac1_tau = ts_test.ktau["ac1"]
                    ac1_records.append(
                        {
                            "defo_level": deforate,
                            "variable": test_var,
                            "mean_ac1": mean_ac1,
                            "ac1_tau": ac1_tau,
                            "rain_multiplier" : rain_multiplier,
                        }
                    )
                except Exception as e:
                    print(f"Could not process {test_var} at defo={deforate:.3f}: {e}")

summary_df = pd.DataFrame(ac1_records)
for variable in vars_to_analyze:

    pivot = (
        summary_df[
            summary_df["variable"] == variable
        ]
        .pivot(
            index="rain_multiplier",
            columns="defo_level",
            values="ac1_tau"
        )
    )

    plt.figure(figsize=(9,7))

    plt.imshow(
        pivot,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max()
        ],
        cmap="viridis"
    )

    cbar = plt.colorbar()
    cbar.set_label("Kendall Tau of AC1", fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)

    plt.xlabel("Annual Deforestation (%)", fontsize=18,fontweight='bold')
    plt.ylabel("Rainfall Multiplier", fontsize=18,fontweight='bold')

    plt.title(f"{domain_prefix} {variable} Slowing Surface", fontsize=21,fontweight='bold')
    cslow_output_file = os.path.expanduser(f"~/{domain_prefix}{variable}_cslow.png")
    plt.savefig(cslow_output_file, bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()

## PIECEWISE BREAKPOINT ANALYSIS**

# pick however many breakpoints actually earns its keep instead of assuming 1 -
# compare a couple segment counts and let SSE tell you when adding another
# breakpoint stops helping
def fit_best_pwlf(x, y, max_segments=3):
    best_model, best_breaks, best_score = None, None, np.inf
    for n_segments in range(1, max_segments + 1):
        model_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = model_pwlf.fit(n_segments)
        sse = model_pwlf.ssr
        penalty = n_segments * np.log(len(x))  # bic-ish penalty
        score = sse + penalty
        if score < best_score:
            best_model, best_breaks, best_score = model_pwlf, breaks, score
    return best_model, best_breaks

n_bins = 3
# Create a temporary column in summary_df tagging each row with its bin label
summary_df['rain_bin'] = pd.qcut(summary_df['rain_multiplier'], q=n_bins, labels=[f"Low Rain", f"Mid Rain", f"High Rain"])

print("\nFITTING BINNED PIECEWISE BREAKPOINTS")
print("=" * 60)

for test_var in vars_to_analyze:
    print(f"\n--- Variable: {test_var} ---")
    
    # Filter for the specific ecosystem component
    var_master_df = summary_df[summary_df["variable"] == test_var]
    
    # Iterate through each rainfall bin independently
    for bin_name in var_master_df['rain_bin'].cat.categories:
        var_df = var_master_df[var_master_df['rain_bin'] == bin_name].sort_values("defo_level")
        
        # Aggregate duplicates (if multiple rainfall levels fall into the same bin)
        df_group = var_df[["defo_level", "ac1_tau"]].sort_values("defo_level")       
        x = df_group["defo_level"].values
        y = df_group["ac1_tau"].values
        
        if len(x) < 10: # Safeguard to ensure enough points exist to fit segments
            continue
            
        # Fit the piecewise linear regression to this specific bin
        model_pwlf, breaks = fit_best_pwlf(x, y)
        interior_breaks = breaks[1:-1]
        
        print(f" Rainfall Bin: {bin_name:<10} | Breakpoints at defo levels: {np.round(interior_breaks, 3)}")
        
        # Plotting the individual bin profile
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color="steelblue", s=40, alpha=0.7, label=f"AC1 ({bin_name})")
        
        x_fit = np.linspace(x.min(), x.max(), 300)
        ax.plot(x_fit, model_pwlf.predict(x_fit), color="firebrick", lw=2.5, label="PWLF Fit") # <-- Capitalized
        
        # Loop through breaks and label the first one to avoid duplicate legend entries
        for i, b in enumerate(interior_breaks):
            ax.axvline(b, color="black", linestyle="--", alpha=0.6, label="Breakpoint" if i == 0 else "")
        
        ax.set_title(f" {domain_prefix} AC1 vs Deforestation | {test_var} ({bin_name})", fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("Deforestation Level (%)", fontsize=14)
        ax.set_ylabel("Kendall Tau (AC1 trend)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=False)
        binoutput_file = os.path.expanduser(f"~/{test_var}{domain_prefix}{bin_name}binnedAC1.png")
        plt.savefig(binoutput_file, bbox_inches="tight", dpi=300)
        plt.show()
        plt.close(fig)
import glob
#plots everythinggg in the run
sim_dir = os.path.expanduser("~/workflowsREU/nnsims")
plot_var = "plantC" # Change to featured variable (e.g., plantC, soilC, LAI)

# Gather and sort saved simulation sweep runs
file_pattern = os.path.join(sim_dir, f"{domain_prefix}simulation_*_{howmanyyears}years.csv")
all_files = glob.glob(file_pattern)

if len(all_files) == 0:
    print("No simulation CSV files discovered. Check directory paths.")
else:
    # Setup maximum stable layout figure dimension
    fig, ax = plt.subplots(figsize=(30, 18), dpi=300)
    
    # Custom color palette gradient,  this looks cool for sweep
    cmap = cm.get_cmap("viridis_r")
    norm = mcolors.Normalize(vmin=0.1, vmax=5.0)
    
    # parse this to make sure it runs correctly 
    parsed_runs = []
    for f in all_files:
        filename = os.path.basename(f)
        try:
            defo_val = float(filename.split("_")[1])
            parsed_runs.append((defo_val, f))
        except ValueError:
            continue
    parsed_runs.sort()
    
    
    
    # Plot each sweep run
    for defo_val, f_path in parsed_runs:
        df_run = pd.read_csv(f_path)
        line_color = cmap(norm(defo_val))
        
        # Emphasize bounding brackets (0.1% and 5.0%)- change to 3 for amzn
        line_alpha = 0.85 if defo_val in [0.1, 5.0] else 0.35
        line_width = 5.0 if defo_val in [0.1, 5.0] else 2.0
        
        ax.plot(
            df_run["Year"], 
            df_run[plot_var], 
            color=line_color, 
            alpha=line_alpha, 
            linewidth=line_width
        )
        
    # Styling and custom typography scaling
    ax.set_title(f"Continuous Emulator Sensitivity Analysis: {plot_var} Cascade", fontsize=38, fontweight="bold", pad=35)
    ax.set_xlabel("Years Into Simulation Simulation", fontsize=26, labelpad=22)
    ax.set_ylabel(f"Simulated Metric State [{plot_var}]", fontsize=26, labelpad=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    
    for border in ["top", "right"]:
        ax.spines[border].set_visible(False)
        
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, aspect=28, shrink=0.85)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_ticks([0.1, 5.0])
    cbar.set_ticklabels(["0.0% Scenario", "5.0% Scenario"])
    cbar.ax.set_ylabel("Applied Deforestation Forcing Rate", rotation=270, labelpad=45, fontsize=24, fontweight="semibold")
    
    plt.tight_layout()
    poster_output_file = os.path.expanduser(f"~/workflowsREU/nnsims/MASSIVE_poster_{plot_var}_sweep.png")
    plt.savefig(poster_output_file, bbox_inches="tight", dpi=300)
    plt.show()

from scipy.optimize import brentq
## importing an optimization script to be able to find when replacement value is lost 
howmanyyears = 80  # extended so both comparison windows exist

def window_slope(trace, start, end):
    # fits a straight line to a window of the trajectory and returns its slope
    seg = np.array(trace[start:end])
    x = np.arange(len(seg))
    return np.polyfit(x, seg, 1)[0]

def get_trend_shift(deforate, sample_idx=1):
    # Retrieve indices for dynamic rainfall and transpiration updates
    transp_idx = biomeinput.index("Transp")
    plantc_idx = biomeinput.index("GPP")
    rain_idx = forceinput.index("RAIN")
    defor_idx = forceinput.index("deforestation_pct")
    
    current_biome_state = X_numpy[sample_idx, len(forceinput):].copy()
    prev_transp = current_biome_state[transp_idx]
    
    plantC_trace = [] #u can change the variable name for anything
    
    with torch.no_grad():
        for year in range(howmanyyears):
            # Extract baseline forcing features (TEMP, Swdown, PRESSURE, RAIN, RH, WIND, defor)
            force_row = X_numpy[sample_idx + year, :len(forceinput)].copy()
            
            # 1. Update dynamic precipitation using my r 
            P_external_t = force_row[rain_idx]  # baseline climate forcing
            force_row[rain_idx] = P_external_t + (r_recycling * prev_transp)
            force_row[defor_idx] = deforate
            
            # Concatenate non-spatial forcing + biome state (No lat/lon)
            input_t = np.hstack([force_row, current_biome_state])
            
            #Scale and predict next state
            input_scaled = scaler_X.transform(input_t.reshape(1, -1))
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
            pred = model(input_tensor)
            
            current_biome_state = scaler_Y.inverse_transform(pred.cpu().numpy())[0]
            
            # 5. Extract updated Transp for step t+1 & track plantC
            prev_transp = current_biome_state[transp_idx]
            plantC_trace.append(current_biome_state[plantc_idx])

    slope_early = window_slope(plantC_trace, 10, 40)
    slope_late = window_slope(plantC_trace, 40, 70)
    
    return slope_late - slope_early

# dense local sweep around the suspected break zone
suspected_center = 2.0
window = 0.5
dense_range = np.linspace(suspected_center - window, suspected_center + window, 60)

records = []
for deforate in dense_range:
    shift = get_trend_shift(deforate)
    records.append({"defo_level": deforate, "trend_shift": shift})

refine_df = pd.DataFrame(records).sort_values("defo_level").reset_index(drop=True)

# find bracket where trend shift crosses zero, then solve for the exact point
sign_changes = np.where(np.diff(np.sign(refine_df["trend_shift"])))[0]
for idx in sign_changes:
    p_low = refine_df["defo_level"].iloc[idx]
    p_high = refine_df["defo_level"].iloc[idx + 1]
    p_star = brentq(get_trend_shift, p_low, p_high, xtol=1e-4)
    print(f"threshold found at defo={p_star:.4f}%")

# plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color="gray", linestyle="--")
ax.scatter(refine_df["defo_level"], refine_df["trend_shift"], color="steelblue")
if len(sign_changes) > 0:
    ax.axvline(p_star, color="firebrick", linestyle="--", label=f"p* = {p_star:.4f}%")
ax.set_xlabel("Deforestation Level")
ax.set_ylabel("Change of Slope (80 Years)")
ax.set_title(f"{domain_prefix} GPP Slope Threshold")
ax.legend()
plt.tight_layout()
plt.show()
# dense sweep of very small deforestation values
small_range = np.linspace(0.001, 0.1, 80)

records = []
for deforate in small_range:
    shift = get_trend_shift(deforate)
    records.append({"defo_level": deforate, "trend_shift": shift})

refine_df = pd.DataFrame(records).sort_values("defo_level").reset_index(drop=True)

# plot it plain, just to look
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color="gray", linestyle="--")
ax.scatter(refine_df["defo_level"], refine_df["trend_shift"], color="steelblue", s=25)
ax.set_xlabel("Deforestation Level")
ax.set_ylabel("Change of Slope Over 80 Years")
ax.set_title(f"{domain_prefix.capitalize()}: fine sweep near small removal rates")
plt.tight_layout()
#plt.savefig("~/trend_shift.png", dpi=300, bbox_inches="tight")
plt.show()