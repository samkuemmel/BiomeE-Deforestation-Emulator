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
#paths
LOCAL_TMP = os.path.abspath("./amazon_work")
master_df = pd.read_csv(os.path.join(LOCAL_TMP, "ammaster_training_datanoprcp.csv")) # need TO CHANGE BETWEEN CERRADO AND AMAZON
master_df.columns = master_df.columns.str.strip()
#remove the device = device call for X and Y if on cpu 
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

forceinput = ["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND", "deforestation_pct", "lat", "lon"]
biomeinput = ["GPP", "LAI", "treeCA", "plantC", "soilC", "rootC", "mu", "Transp", "Indv"]

X_list = []
Y_list = []

# Group by lat lon and deforestation scenario
grouped = master_df.groupby(["lat", "lon", "deforestation_pct"])
for g_keys, group in grouped:
    # Sort by year to ensure chronology
    group = group.sort_values("time")

    if len(group) < 2:
        continue # Skip if we don't have at least 2 consecutive years for this point

    # Inputs: Forcing + Current Biome State at time t
    inputs_t = np.hstack([group[forceinput].iloc[:-1].values, group[biomeinput].iloc[:-1].values])
    # Target: Biome State at time t+1
    targets_t1 = group[biomeinput].iloc[1:].values

    X_list.append(inputs_t)
    Y_list.append(targets_t1)

#LONG WAY TO DO THE TIME STEPPING #targ = df["GPP"] # can be whatever column#l= len(outdf)#thisstep = []#nextstep = []#for x in range(l-1):
    #pulling values from each col in dataset#    flist= df.loc[x,forceinput].values#    blist = outdf.loc[x,biomeinput].values#    combo = np.concatenate([flist,blist])#    thisstep.append(combo)
    # this is doing it a time step ahead to enable the predicting
    #    blist2= outdf.loc[x+1, biomeinput].values#    nextstep.append(blist2)#the vectorized time step#its like if you buy two calendars and align one a day ahead
    # THE REPLACEMENT:# Combine all the grouped timeline arrays into single master matrices
X_numpy = np.vstack(X_list)
Y_numpy = np.vstack(Y_list)
Y_numpy_log = np.log1p(np.maximum(0, Y_numpy))
#turns into tensors, gets size for next step#scales them first
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_numpy)
Y_scaled = scaler_Y.fit_transform(Y_numpy_log)
X = torch.tensor(X_scaled, dtype=torch.float32, device = device) # remove the device = device if on cpu
Y = torch.tensor(Y_scaled, dtype=torch.float32, device = device)
xdim = X.shape[1]
ydim = Y.shape[1]
criterion = nn.MSELoss()
#input prob deforestation forcing, output probably gpp/npp/whatever one col at a time?
# DataLoader, replace these with x and y cols
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=4096, shuffle=True,)

# Neural Network, in and out features account for diff sizesclass SimpleNN(nn.Module):
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

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
model = SimpleNN(in_features=xdim, out_features=ydim).to(device)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []
fold_r2s = []
batchsize = 4096
print(f"Running 5-Fold Cross-Validation across {len(X)} samples...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_v, Y_v = X[val_idx], Y[val_idx]
    cv_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batchsize, shuffle=True)
    cv_model = SimpleNN(in_features=xdim, out_features=ydim).to(device)
    cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=0.001,weight_decay=1e-5)
    #train loop
    cv_model.train()
    nsamp = len (X_tr)
    for epoch in range(100):
        ##GPU version
        rperm = torch.randperm(nsamp, device=device)
        X_shuffled = X_tr[rperm]
        Y_shuffled = Y_tr[rperm]
        for i in range(0, nsamp, batchsize):
            bx = X_shuffled[i:i + batchsize]
            by = Y_shuffled[i:i + batchsize]
    
            loss = criterion(cv_model(bx), by)
            cv_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cv_model.parameters(), max_norm=0.5)
            cv_optimizer.step()
        ##CPU version
       # for bx, by in cv_loader:
        #    bx, by = bx.to(device), by.to(device)
         #   loss = criterion(cv_model(bx), by)
          #  cv_optimizer.zero_grad()
           # loss.backward()
            #torch.nn.utils.clip_grad_norm_(cv_model.parameters(), max_norm=0.5)
            #cv_optimizer.step()
    # validating loop
    cv_model.eval()
    with torch.no_grad():
        # Convert fold validation slice to tensors
        X_v_tensor = X_v.to(device)
        Y_v_tensor = Y_v.to(device)
        #validating the mse loss cause i use mse
        v_loss = criterion(cv_model(X_v_tensor), Y_v_tensor).item()
        fold_losses.append(v_loss)
        # validating cv prediction
        val_preds_scaled = cv_model(X_v_tensor).cpu().numpy()
        val_true_scaled = Y_v_tensor.cpu().numpy()
        # its inverting the scaling, then doing a r^2
        val_preds_real = np.expm1(scaler_Y.inverse_transform(val_preds_scaled))
        val_true_real = np.expm1(scaler_Y.inverse_transform(val_true_scaled))
        fold_r2 = r2_score(val_true_real, val_preds_real, multioutput="uniform_average")
        fold_r2s.append(fold_r2)
        print(f"  Fold {fold + 1}/5 | Val Loss: {v_loss:.4f} | Val R²: {fold_r2:.4f}")
print(f"\nMean CV Loss: {np.mean(fold_losses):.4f} (+/- {np.std(fold_losses):.4f})")
print(f"Mean CV R²:   {np.mean(fold_r2s):.4f} (+/- {np.std(fold_r2s):.4f})")

# Loss and optimizercriterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
num_epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
# Training loop, running batches of x and y num_epochs = 150for epoch in range(num_epochs):

for epoch in range(num_epochs):
    model.train()
    rperm = torch.randperm(nsamp, device=device)
    X_shuf = X[rperm]
    Y_shuf = Y[rperm]
    for i in range(0, nsamp, batchsize):
        bx = X_shuf[i:i + batchsize]
        by = Y_shuf[i:i + batchsize]
    
        loss = criterion(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Testxtest = torch.randn(1,xdim).to(device)
xtest = torch.randn(1, xdim).to(device)
model.eval() # calling the neural network# predicts one with the actual numbers cause i scaled them earlier with torch.no_grad():
with torch.no_grad():
    prediction = model(xtest)
    # unscale
    prediction_numpy = prediction.cpu().numpy()
    pred_unscaled = scaler_Y.inverse_transform(prediction_numpy)
    real_predictions = np.expm1(pred_unscaled)
    real_predictions = np.maximum(0.0, real_predictions)

    print("predicted biomevalues:")
    for name, value in zip(biomeinput, real_predictions[0]):
        print(f"{name}: {value:.4f}") #WEIGHTS SAVING. EXTREMELY IMPORTANT! # change this for CERRADO/Amazonimport joblib import ossavepath= os.path.expanduser("~/workflowsREU/models")

# change this for CERRADO/Amazon
savepath = os.path.expanduser("~/workflowsREU/models")
usepath = os.path.join(savepath, "AmazonSpace_nnmodel")
torch.save(model.state_dict(), usepath)
scalexpath = os.path.join(savepath, "AmazonscalerSpace_X.joblib")
scaleypath = os.path.join(savepath, "AmazonscalerSpace_Y.joblib")
joblib.dump(scaler_X, scalexpath)
joblib.dump(scaler_Y, scaleypath)