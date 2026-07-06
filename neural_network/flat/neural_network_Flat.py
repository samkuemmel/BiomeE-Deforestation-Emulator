import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
#take this from the aggregating code
# load data & prepare sequences (non-spatial)
master_df = pd.read_csv("~/amazon_work/amazon_spatial_averaged.csv")
master_df.columns = master_df.columns.str.strip()

# device & seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# input definitions (non-spatial)
forceinput = ["TEMP", "Swdown", "PRESSURE", "RAIN", "RH", "WIND", "deforestation_pct"]
biomeinput = ["GPP", "LAI", "treeCA", "plantC", "soilC", "rootC", "mu", "Transp", "Indv"]

X_list = []
Y_list = []

# group by location & deforestation scenario to form consecutive time steps
grouped = master_df.groupby(["deforestation_pct", "precip_scale"])
for g_keys, group in grouped:
    group = group.sort_values("time")
    
    if len(group) < 2:
        continue
        
    inputs_t = np.hstack([group[forceinput].iloc[:-1].values, group[biomeinput].iloc[:-1].values])
    targets_t1 = group[biomeinput].iloc[1:].values
    
    X_list.append(inputs_t)
    Y_list.append(targets_t1)

X_numpy = np.vstack(X_list)
Y_numpy = np.vstack(Y_list)

# scale data directly
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_numpy)
Y_scaled = scaler_Y.fit_transform(Y_numpy)

X = torch.tensor(X_scaled, dtype=torch.float32)
Y = torch.tensor(Y_scaled, dtype=torch.float32)

xdim = X.shape[1]
ydim = Y.shape[1]

# model definition
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

# cross validation
criterion = nn.MSELoss()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []
fold_r2s = []

print(f"Running 5-Fold Cross-Validation across {len(X)} samples...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_v, Y_v = X[val_idx], Y[val_idx]
    
    cv_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=32, shuffle=True)
    cv_model = SimpleNN(in_features=xdim, out_features=ydim).to(device)
    cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=0.001)
    
    # training loop
    cv_model.train()
    for epoch in range(300):  
        for bx, by in cv_loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(cv_model(bx), by)
            cv_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cv_model.parameters(), max_norm=0.5)
            cv_optimizer.step()
            
    # validation loop
    cv_model.eval()
    with torch.no_grad():
        X_v_tensor = X_v.to(device)
        Y_v_tensor = Y_v.to(device)

        # 1. Validation MSE Loss
        v_loss = criterion(cv_model(X_v_tensor), Y_v_tensor).item()
        fold_losses.append(v_loss)

        # 2. Validation Predictions using cv_model
        val_preds_scaled = cv_model(X_v_tensor).cpu().numpy()
        val_true_scaled = Y_v_tensor.cpu().numpy()

        # 3. Inverse-transform to physical units and compute R²
        val_preds_real = scaler_Y.inverse_transform(val_preds_scaled)
        val_true_real = scaler_Y.inverse_transform(val_true_scaled)

        fold_r2 = r2_score(val_true_real, val_preds_real, multioutput="uniform_average")
        fold_r2s.append(fold_r2)

        print(f"  Fold {fold + 1}/5 | Val Loss: {v_loss:.4f} | Val R²: {fold_r2:.4f}")

print(f"\nMean CV Loss: {np.mean(fold_losses):.4f} (+/- {np.std(fold_losses):.4f})")
print(f"Mean CV R²:   {np.mean(fold_r2s):.4f} (+/- {np.std(fold_r2s):.4f})")
# final model training with scheduler & clipping
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleNN(in_features=xdim, out_features=ydim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

num_epochs = 300
print("\nTraining Final Model...")
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# inference & evaluation
xtest = torch.randn(1, xdim).to(device)
model.eval()

with torch.no_grad():
    prediction = model(xtest)
    prediction_numpy = prediction.cpu().numpy()
    
    real_predictions = scaler_Y.inverse_transform(prediction_numpy)
    
    print("\nPredicted biome values:")
    for name, value in zip(biomeinput, real_predictions[0]):
        print(f"{name}: {value:.4f}")

# save weights & scalers
savepath = os.path.expanduser("~/workflowsREU/models")
os.makedirs(savepath, exist_ok=True)

usepath = os.path.join(savepath, "AmazonFlat_nnmodel")
scalexpath = os.path.join(savepath, "AmazonscalerFlat_X.joblib")
scaleypath = os.path.join(savepath, "AmazonscalerFlat_Y.joblib")

torch.save(model.state_dict(), usepath)
joblib.dump(scaler_X, scalexpath)
joblib.dump(scaler_Y, scaleypath)
print("\nModel and scalers saved successfully.")