import os
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# BIOME TOGGLE 

st.set_page_config(
    page_title="Ecosystem Neural Network Forecaster Under Deforestation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.markdown("## 🌍 Target Biome Matrix")
selected_biome = st.sidebar.radio(
    "Choose evaluation ecosystem:",
    options=["Cerrado", "Amazon"],
    index=0,
    horizontal=True
)
import base64
import os

#  Define config settings and absolute file paths based on the biome
# Update this with whatever image you want
if selected_biome == "Amazon":
    bg_image_path = "/Users/samkuemmel/Downloads/InteractiveBiome/AmazonForest.avif"
    primary_color = "#15803D"
    accent_color = "#22C55E"
    light_accent = "#F0FDF4"
    sub_title_text = "Predicting Responses Across Ecosystem Variables"
    mime_type = "image/avif"
else:
    bg_image_path = "/Users/samkuemmel/Downloads/InteractiveBiome/CerradoImage.jpg"
    primary_color = "#B45309"
    accent_color = "#F59E0B"
    light_accent = "#FEF3C7"
    sub_title_text = "Cerrado Savanna Ecosystem Simulator"
    mime_type = "image/jpeg"

# Read the local file 
try:
    with open(bg_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_css_value = f"url('data:{mime_type};base64,{encoded_string}')"
except FileNotFoundError:
    bg_css_value = "none"

st.markdown(f"""
    <style>
    /* 1. Eliminate the invisible header block height completely */
    header[data-testid="stHeader"] {{
        height: 0px !important;
        background: transparent !important;
    }}
    
    /* 2. Strip all top padding and margins from the main page body content container */
    .block-container {{
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }}
    
    [data-testid="stMainBlockContainer"] {{
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }}

    /* 3. Your header banner adjustments */
    .header-banner {{
        background-image: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), {bg_css_value};
        background-size: cover;
        background-position: center;
        padding: 2.5rem;
        border-radius: 0.5rem;
        border-bottom: 4px solid {accent_color};
        margin-top: 0rem !important; /* Locks it to the absolute upper lip of the viewport */
        margin-bottom: 2rem;
    }}
    .main-title {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {primary_color};
        margin: 0;
    }}
    </style>
""", unsafe_allow_html=True)

# Create the title block inside the styled container banner
st.markdown(f"""
    <div class='header-banner'>
        <div class='main-title'>{selected_biome} BiomeE Deforestation Emulator</div>
        <div style='font-size: 1.1rem; color: #4B5563;'>{sub_title_text}</div>
    </div>
""", unsafe_allow_html=True)


# Loading up the NN
# my local paths are in here, you will need to update with your paths from your model run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_resources(biome):
    #  Swap paths out here based on toggle
    if biome == "Cerrado":
        scaler_X_path = os.path.expanduser("~/Downloads/InteractiveBiome/Cerradoscaler_x.joblib")
        scaler_Y_path = os.path.expanduser("~/Downloads/InteractiveBiome/Cerradoscaler_y.joblib")
        modelpath = os.path.expanduser("~/Downloads/InteractiveBiome/cerrado_nnmodel")
    else: 
        # update amazon model paths
        scaler_X_path = os.path.expanduser("~/Downloads/InteractiveBiome/Amazonscaler_x.joblib") 
        scaler_Y_path = os.path.expanduser("~/Downloads/InteractiveBiome/Amazonscaler_y.joblib")
        modelpath = os.path.expanduser("~/Downloads/InteractiveBiome/amazon_nnmodel")
    
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)
    
    model = SimpleNN(in_features=scaler_X.n_features_in_, out_features=scaler_Y.n_features_in_).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()
    
    return scaler_X, scaler_Y, model

@st.cache_data
def load_historical_data(biome):
    if biome == "Cerrado":
        path = os.path.expanduser("~/Downloads/InteractiveBiome/crmaster_training_data.csv")
    else:
        path = os.path.expanduser("~/Downloads/InteractiveBiome/ammaster_training_data.csv")
    
    return pd.read_csv(path)

# Run load based on toggle choice
try:
    scaler_X, scaler_Y, model = load_resources(selected_biome)
    df_historical = load_historical_data(selected_biome)
except Exception as e:
    st.error(f"Initialization Error loading {selected_biome} files: {e}")
    st.info("Check that file paths inside the `load_resources` if/else loop match your local file layout structure.")
    st.stop()

# Adjusting the simulation controls
st.sidebar.markdown("---")
st.sidebar.markdown("## Simulation Controls")

st.sidebar.caption("⚠️ Rates above 2% in Cerrado and 1% in Amazon currently provide nonsensical data. Longer timeframes also increase risk of nonsensical data, such as a sharp turnaround many years in to a decline")
deforate = st.sidebar.slider("🌳 Deforestation Rate (%)", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
howmanyyears = st.sidebar.slider("⏳ Horizon Timeline (Years)", min_value=10, max_value=100, value=50, step=5)

with st.sidebar.expander("⚙️ Advanced Setup"):
    sample_idx = st.number_input("Historical Row Baseline Index", min_value=0, max_value=len(df_historical)-howmanyyears-1, value=100)

# Build feature arrays
forceinput = ["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND", "deforestation_pct"]
biomeinput = ["GPP", "LAI", "treeCA", "plantC", "soilC", "rootC", "mu", "Transp", "Indv"]

X_numpy = df_historical[forceinput + biomeinput].values
initial_row = X_numpy[sample_idx].copy()
current_biome_state = initial_row[7:].copy()

##running the NN
simulation_history = []

with torch.no_grad():
    for year in range(howmanyyears):
        actual_row_this_year = X_numpy[sample_idx + year]
        weather_this_year = actual_row_this_year[:6]
        
        input_t = np.hstack([weather_this_year, [deforate], current_biome_state])
        input_scaled = scaler_X.transform(input_t.reshape(1, -1))
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        
        pred_scaled = model(input_tensor)
        next_biome_state = scaler_Y.inverse_transform(pred_scaled.cpu().numpy())[0]
        
        current_biome_state = next_biome_state
        
        history_entry = {"Year": year + 1, "Deforestation_Pct": deforate}
        for name, val in zip(biomeinput, next_biome_state):
            history_entry[name] = val
        simulation_history.append(history_entry)

df_sim = pd.DataFrame(simulation_history)

# Save the data matrix
output_dir = os.path.expanduser("~/workflowsREU/nnsims/")
os.makedirs(output_dir, exist_ok=True)
filename = f"{selected_biome.lower()}simulation_{deforate}_{howmanyyears}years.csv"
df_sim.to_csv(os.path.join(output_dir, filename), index=False)

#layout framework
st.markdown("""
    <style>
    /* Force the metric column container to drop its bottom spacing */
    [data-testid="stHorizontalBlock"] {
        margin-bottom: -20px !important;
    }
    
    /* Optional: Shrink the inner padding of the metrics themselves */
    [data-testid="stMetricValue"] {
        margin-bottom: -10px !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown(f"<div class='card-container'><strong>Active Evaluation State:</strong> Mapping a <b>{deforate}%</b> annual deforestation rate across a <b>{howmanyyears} year</b> projection inside the <b>{selected_biome} ecosystem profile</b>.</div>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: -30px;'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Target Region Focus", value=selected_biome)
with col2:
    st.metric(label="Deforestation Rate", value=f"{deforate}%")
with col3:
    final_gpp = df_sim["GPP"].iloc[-1]
    delta_gpp = final_gpp - df_sim["GPP"].iloc[0]
    st.metric(label="Terminal GPP", value=f"{final_gpp:.2f}", delta=f"{delta_gpp:.2f} delta")
with col4:
    final_lai = df_sim["LAI"].iloc[-1]
    st.metric(label="Terminal LAI", value=f"{final_lai:.2f}")
st.markdown("</div>", unsafe_allow_html=True) 
st.markdown("<hr style='margin-top: -3px; margin-bottom: 10px; border: 0; border-top: 1px solid rgba(49, 51, 63, 0.2);'>", unsafe_allow_html=True)
# Organized Tabs layout
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "💾 Export Data Matrix"])

with tab1:
    st.subheader("Predictive Biome Trajectories")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("<br>", unsafe_allow_html=True)
        selected_metric = st.selectbox("Select Target Attribute to Map:", biomeinput, index=0)
        
        # Add a slider to control sliding
        window_size = st.slider("Smoothing Window (Years Rolling Mean)", min_value=1, max_value=10, value=5)
        
        if selected_biome == "Amazon":
            sub_image_path = "/Users/samkuemmel/Downloads/InteractiveBiome/AmazonClip.png" # 🔔 Put your Amazon image path here
            sub_mime = "image/png"
        else:
            sub_image_path = "/Users/samkuemmel/Downloads/InteractiveBiome/CerradoClip.png" # 🔔 Put your Cerrado image path here
            sub_mime = "image/png" # Change to image/jpeg if your file is a .jpg

        # Read image
        try:
            with open(sub_image_path, "rb") as sub_img_file:
                sub_encoded = base64.b64encode(sub_img_file.read()).decode()
            
            
            st.markdown(
                f'<img src="data:{sub_mime};base64,{sub_encoded}" style="width: 200px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: block; margin: 0 auto;">',
                unsafe_allow_html=True
            )
        except FileNotFoundError:
            st.caption("ℹ️ *Ecosystem graphic asset slot placeholder (verify local image path link)*")
    with c2:
        
        df_plot = df_sim.copy()
        df_plot[selected_metric] = df_plot[selected_metric].rolling(window=window_size, min_periods=1).mean()
        
        # Build line chart
        fig = px.line(
            df_plot, 
            x="Year", 
            y=selected_metric,
            title=f"Projected {selected_metric} Evolution ({window_size}-Year Rolling Mean)",
            labels={selected_metric: f"Smoothed Value ({selected_metric})", "Year": "Timeline Year"},
            template="plotly_white"
        )
        fig.update_traces(line=dict(color=primary_color, width=3), mode='lines+markers')
        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.subheader("Raw Data Summary")
    st.dataframe(df_sim.style.format(precision=4), use_container_width=True)
    
    d_col1, d_col2 = st.columns([1, 2])
    with d_col1:
        csv_bytes = df_sim.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {selected_biome} Data (CSV)",
            data=csv_bytes,
            file_name=f"{selected_biome.lower()}_simulation_{deforate}pct.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d_col2:
        st.info(f"💾 **Local Safe Point:** Matrix logged automatically on background storage engine: `{os.path.join(output_dir, filename)}`")