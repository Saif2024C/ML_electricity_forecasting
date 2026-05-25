#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib          # pip install joblib if missing
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


# In[2]:


df = pd.read_csv("household_power_consumption.txt", sep=";",low_memory=False)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df.isnull().sum())


# In[6]:


print((df.isnull().mean() * 100).round(2))


# ## Pre-processing 

# In[7]:


start_date = '2007-01-01'
end_date   = '2008-01-01'


# In[8]:


plt.style.use('default')

# =========================
# 2. Create Datetime Index
# =========================

def Date_time(df,start_date,end_date):
    
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],format='%d/%m/%Y %H:%M:%S')
    
    df = df.set_index('Datetime')
    df = df.drop(columns=['Date', 'Time'])
    
    #CRITICAL: convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Interpolate missing values using time index
    df = df.ffill()
    
    # Cyclical time features
    hours = df.index.hour
    dow = df.index.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)
    
    window = df.loc[start_date:end_date]     
    return df,window


# In[9]:


df_dt, window = Date_time(df,start_date,end_date)
TARGET = 'Global_active_power'  # change if needed

plt.figure(figsize=(15, 5))
plt.plot(window.index, window[TARGET])
plt.title(f'{TARGET} ({start_date} to {end_date})')
plt.xlabel('Time')
plt.ylabel(TARGET)
plt.show()


# In[10]:


# ===============================
# Hourly & Daily Multiplot
# ===============================

# Resample
hourly = df_dt.resample('1H').mean()
daily  = df_dt.resample('1D').mean()
fig, axes = plt.subplots( nrows=1, ncols=2, figsize=(16, 6),sharey=True)

# ---- Hourly plot ----
axes[0].plot( hourly.index, hourly[TARGET], label='Hourly Mean', linewidth=0.8)
axes[0].set_title(f'Hourly Average {TARGET}', fontsize=12)
axes[0].set_xlabel('Time')
axes[0].set_ylabel(TARGET)
axes[0].grid(True, which='both', linestyle='--', alpha=0.6)
axes[0].legend()

# ---- Daily plot ----
axes[1].plot(daily.index,daily[TARGET],label='Daily Mean',linewidth=1.2)
axes[1].set_title(f'Daily Average {TARGET}', fontsize=12)
axes[1].set_xlabel('Date')
axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
axes[1].legend()

# ---- Global title ----
fig.suptitle(f'{TARGET}: Hourly vs Daily Aggregation',fontsize=14,y=1.02)
plt.tight_layout()
plt.show()


# In[11]:


# -----------------------------
# Additions: splitting, scaling, windowing, dataset
# -----------------------------


# ---------- 1) Chronological split ----------
def chrono_split(df, train_frac=0.70, val_frac=0.15):
    """
    Chronological split. Returns train, val, test DataFrames.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_train + n_val].copy()
    test  = df.iloc[n_train + n_val:].copy()
    return train, val, test


# ---------- 2) Scaling (fit on train only) ----------
def fit_scalers(train_df, feature_cols, target_col):
    """
    Fit StandardScaler for X (features) and y (target).
    Returns fitted (x_scaler, y_scaler).
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[feature_cols])
    y_scaler.fit(train_df[[target_col]])

    return x_scaler, y_scaler


def apply_scalers(df, feature_cols, target_col, x_scaler, y_scaler):
    """
    Apply fitted scalers to a dataframe; returns numpy arrays (X, y).
    X: shape (n, F), y: shape (n, 1)
    """
    X = x_scaler.transform(df[feature_cols])
    y = y_scaler.transform(df[[target_col]])
    return X.astype(np.float32), y.astype(np.float32)


# ---------- 3) Windowing ----------
def make_windows(X, y, T=168, H=24):
    """
    X: np.array (n, F)
    y: np.array (n, 1)
    Returns Xw: (num_windows, T, F) and Yw: (num_windows, H)
    """
    n = X.shape[0]
    num = n - T - H + 1
    if num <= 0:
        raise ValueError(f"Not enough data for windows: n={n}, need at least T+H={T+H}")
    F = X.shape[1]
    Xw = np.zeros((num, T, F), dtype=np.float32)
    Yw = np.zeros((num, H), dtype=np.float32)
    y_flat = y.reshape(-1)  # (n,)
    for i in range(num):
        Xw[i] = X[i:i+T]
        Yw[i] = y_flat[i+T:i+T+H]
    return Xw, Yw


# ---------- 4) Save / Load scalers ----------
def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)


# ---------- 5) PyTorch Dataset wrapper (optional) ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, X_windows, Y_windows):
        # X_windows: (N, T, F), Y_windows: (N, H)
        self.X = torch.from_numpy(X_windows)          # float32
        self.Y = torch.from_numpy(Y_windows)          # float32
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# -----------------------------
# Example: full pipeline usage (hourly)
# -----------------------------
# Use the df_dt and window you already produce with Date_time(...)
# Choose whether to resample the full df_dt or the window. We'll use `window` (matches your plots)
# You can change resample_rule to "1D" for daily processing.

# params
TARGET = 'Global_active_power'
RESAMPLE_RULE = '1H'   # '1H' for hourly, '1D' for daily
T_hourly = 168
H_hourly = 24
T_daily = 30
H_daily = 7

# 1) resample the window to desired frequency (keeps same start/end)
proc_df = window.resample(RESAMPLE_RULE).mean()   # window is datetime-indexed and cleaned

# 2) pick feature columns (all numeric columns except TARGET)
# ensure cyclical features are present (they were created in Date_time on df_dt)
feature_cols = [c for c in proc_df.columns if c != TARGET]

print(feature_cols)
# 3) chrono-split
train_df, val_df, test_df = chrono_split(proc_df, train_frac=0.70, val_frac=0.15)

# 4) fit scalers on train only
x_scaler, y_scaler = fit_scalers(train_df, feature_cols, TARGET)

# Save scalers for later (optional)
save_scaler(x_scaler, 'x_scaler.joblib')
save_scaler(y_scaler, 'y_scaler.joblib')

# 5) apply scalers
X_train_raw, y_train_raw = apply_scalers(train_df, feature_cols, TARGET, x_scaler, y_scaler)
X_val_raw,   y_val_raw   = apply_scalers(val_df, feature_cols, TARGET, x_scaler, y_scaler)
X_test_raw,  y_test_raw  = apply_scalers(test_df, feature_cols, TARGET, x_scaler, y_scaler)

# 6) windowing: choose T/H based on RESAMPLE_RULE
if RESAMPLE_RULE == '1H':
    T, H = T_hourly, H_hourly
else:
    T, H = T_daily, H_daily

X_train_w, Y_train_w = make_windows(X_train_raw, y_train_raw, T=T, H=H)
X_val_w,   Y_val_w   = make_windows(X_val_raw,   y_val_raw,   T=T, H=H)
X_test_w,  Y_test_w  = make_windows(X_test_raw,  y_test_raw,  T=T, H=H)

print("X_train_w:", X_train_w.shape, "Y_train_w:", Y_train_w.shape)
print("X_val_w:  ", X_val_w.shape,   "Y_val_w:",   Y_val_w.shape)
print("X_test_w: ", X_test_w.shape,  "Y_test_w:",  Y_test_w.shape)

# 7) PyTorch dataloaders (optional)
train_ds = TimeSeriesDataset(X_train_w, Y_train_w)
val_ds   = TimeSeriesDataset(X_val_w, Y_val_w)
test_ds  = TimeSeriesDataset(X_test_w, Y_test_w)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# Now train with train_loader; inputs are (B, T, F), targets (B, H)


# # LSTM

# In[12]:


class LSTMForecaster(nn.Module):
    """
    Many-to-many (multi-step) forecasting:
    Input:  (B, T, F)
    Output: (B, H)
    """
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, horizon=24):
        super().__init__()
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, hidden)
        last = out[:, -1, :]           # last hidden state: (B, hidden)
        yhat = self.head(last)         # (B, H)
        return yhat


# In[13]:


def to_numpy(x):
    return x.detach().cpu().numpy()

def inverse_scale_y(y_scaled, y_scaler):
    """
    y_scaled: (N, H) numpy
    y_scaler was fit on shape (N, 1); apply per-element inverse by reshaping.
    """
    N, H = y_scaled.shape
    y_inv = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(N, H)
    return y_inv

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# In[14]:


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, Y in loader:
        X = X.to(device)   # (B, T, F)
        Y = Y.to(device)   # (B, H)

        optimizer.zero_grad()
        Yhat = model(X)                # (B, H)
        loss = loss_fn(Yhat, Y)        # compare in scaled space
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, y_scaler):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    preds_scaled = []
    trues_scaled = []

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        Yhat = model(X)
        loss = loss_fn(Yhat, Y)

        total_loss += loss.item()
        n_batches += 1

        preds_scaled.append(to_numpy(Yhat))
        trues_scaled.append(to_numpy(Y))

    preds_scaled = np.concatenate(preds_scaled, axis=0)  # (N, H)
    trues_scaled = np.concatenate(trues_scaled, axis=0)  # (N, H)

    # Inverse-scale to original units for metrics
    preds = inverse_scale_y(preds_scaled, y_scaler)
    trues = inverse_scale_y(trues_scaled, y_scaler)

    metrics = {
        "loss_scaled": total_loss / max(n_batches, 1),
        "MAE": mae(trues, preds),
        "RMSE": rmse(trues, preds),
    }
    return metrics


# In[15]:


import copy
import torch.optim as optim

def fit_lstm(
    train_loader,
    val_loader,
    y_scaler,
    n_features,
    horizon=24,
    hidden_size=16,
    num_layers=1,
    dropout=0.35,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()  # train on scaled targets

    best_state = None
    best_val_rmse = float("inf")

    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, y_scaler)

        record = {
            "epoch": epoch,
            "train_loss_scaled": train_loss,
            **val_metrics
        }
        history.append(record)

        # select best by val RMSE (original units)
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss(scaled)={train_loss:.4f} | "
            f"val_RMSE={val_metrics['RMSE']:.4f} | "
            f"val_MAE={val_metrics['MAE']:.4f} | "
        )

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# In[16]:


n_features = X_train_w.shape[2]
horizon = Y_train_w.shape[1]

model, history = fit_lstm(
    train_loader=train_loader,
    val_loader=val_loader,
    y_scaler=y_scaler,
    n_features=n_features,
    horizon=horizon,
    epochs=6,
)

# Final test metrics
test_metrics = evaluate(
    model=model,
    loader=test_loader,
    loss_fn= nn.SmoothL1Loss(),
    device="cuda" if torch.cuda.is_available() else "cpu",
    y_scaler=y_scaler
)

print("TEST METRICS:", test_metrics)


# In[17]:


import matplotlib as mpl

mpl.rcParams.update({
    # Figure
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "figure.facecolor": "white",

    # Fonts (safe academic default)
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,

    # Axes
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,

    # Grid (academic style)
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.5,

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})


# In[71]:


# ====== Paste this cell at the END of your notebook (after you computed test_metrics) ======
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

# helper (you already have similar helpers, but keep local versions to be self-contained)
def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Device detection (use same device you trained on)
device = next(model.parameters()).device if 'model' in globals() else ("cuda" if torch.cuda.is_available() else "cpu")

# 1) Run model on test_loader, collect scaled preds and trues
preds_scaled_list = []
trues_scaled_list = []

model.eval()
with torch.no_grad():
    for Xb, Yb in test_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        Yhat = model(Xb)                   # (B, H) scaled
        preds_scaled_list.append(Yhat.detach().cpu().numpy())
        trues_scaled_list.append(Yb.detach().cpu().numpy())

if len(preds_scaled_list) == 0:
    raise RuntimeError("test_loader yielded no batches. Make sure test_loader is defined and not empty.")

preds_scaled = np.concatenate(preds_scaled_list, axis=0)  # (N, H)
trues_scaled = np.concatenate(trues_scaled_list, axis=0)  # (N, H)

# 2) inverse-scale to original units using your inverse_scale_y function
#    (your notebook already defines inverse_scale_y(y_scaled, y_scaler))
preds = inverse_scale_y(preds_scaled, y_scaler)   # (N, H)
trues = inverse_scale_y(trues_scaled, y_scaler)   # (N, H)

# 3) Global flattened residuals for diagnostics
resid_flat = (trues - preds).reshape(-1)   # true - pred
res = resid_flat - np.mean(resid_flat)

# 4) Basic metrics (on original scale)
mae_val = mean_absolute_error(trues.reshape(-1), preds.reshape(-1))
rmse_val = rmse_np(trues.reshape(-1), preds.reshape(-1))
r2_val = r2_score(trues.reshape(-1), preds.reshape(-1))

print("=== Global metrics (original units) ===")
print(f"N residuals: {res.size}")
print(f"MAE:  {mae_val:.6g}")
print(f"RMSE: {rmse_val:.6g}")
print(f"R² (flattened): {r2_val:.6g}")
print("")

# 5) Fit Normal & Laplace MLE scale params
sigma_hat = np.sqrt(np.mean(res**2))   # normal sigma MLE
b_hat = np.mean(np.abs(res))           # Laplace b MLE

print("=== Residual summary ===")
print(f"Estimated Normal sigma (MLE): {sigma_hat:.6g}")
print(f"Estimated Laplace b (MLE):   {b_hat:.6g}")
print(f"Skewness: {stats.skew(res):.4f}")
print(f"Excess kurtosis (Fisher): {stats.kurtosis(res):.4f}")
print("")

# 6) Plots: histogram + fitted PDFs, QQs, boxplot
x = np.linspace(np.percentile(res,1), np.percentile(res,99), 200)
pdf_norm = stats.norm.pdf(x, loc=0, scale=sigma_hat)
pdf_lap  = stats.laplace.pdf(x, loc=0, scale=b_hat)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.hist(res, bins=80, density=True, alpha=0.6, label='residuals')
plt.plot(x, pdf_norm, lw=2, label=f'Normal PDF (σ={sigma_hat:.3g})')
plt.plot(x, pdf_lap,  lw=2, label=f'Laplace PDF (b={b_hat:.3g})')
plt.title("Residual histogram with fitted PDFs")
plt.legend()

plt.subplot(2,2,2)
stats.probplot(res, dist="norm", plot=plt)
plt.title("QQ-plot vs Normal")

plt.subplot(2,2,3)
qt = stats.laplace.ppf(np.linspace(0.01,0.99,100), loc=0, scale=b_hat)
emp_q = np.quantile(res, np.linspace(0.01,0.99,100))
plt.scatter(qt, emp_q, s=10)
plt.plot([qt.min(), qt.max()], [qt.min(), qt.max()], color='C1')
plt.xlabel('Laplace theoretical quantiles'); plt.ylabel('Residual empirical quantiles')
plt.title('QQ-plot vs Laplace')

plt.subplot(2,2,4)
plt.boxplot(res, vert=False)
plt.title("Residual boxplot")
plt.tight_layout()
plt.show()

# 7) ACF
plt.figure(figsize=(10,3))
plot_acf(res, lags=48, alpha=0.05)
plt.title("ACF of residuals (flattened)")
plt.tight_layout()
plt.show()

# 8) Numeric tests
from scipy import stats as st
skewness = st.skew(res)
kurtosis_excess = st.kurtosis(res)
print("=== Numeric tests ===")
print(f"Skewness = {skewness:.4f}")
print(f"Excess kurtosis = {kurtosis_excess:.4f}")
try:
    sh_stat, sh_p = st.shapiro(res if res.size<=5000 else res[:5000])
    print(f"Shapiro-Wilk: W={sh_stat:.4f}, p={sh_p:.4g}  (use with caution for large N)")
except Exception:
    print("Shapiro-Wilk test skipped (too large sample or not available).")
k2_stat, k2_p = st.normaltest(res)
print(f"D'Agostino's K2: stat={k2_stat:.4f}, p={k2_p:.4g}")
ad_result = st.anderson(res, dist='norm')
print("Anderson-Darling (normal) statistic:", ad_result.statistic)
for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
    print(f"  {sl}% : critical {cv:.3f}")
print("")

# 9) Log-likelihoods & AIC comparison
n = res.size
ll_normal = - (n/2) * np.log(2*np.pi*sigma_hat**2) - (np.sum(res**2) / (2*sigma_hat**2))
ll_laplace = - n * np.log(2*b_hat) - (np.sum(np.abs(res)) / b_hat)
k = 2
aic_norm = 2*k - 2*ll_normal
aic_lap  = 2*k - 2*ll_laplace

print("=== Likelihood comparison ===")
print(f"Log-likelihood (normal)  : {ll_normal:.6g}")
print(f"Log-likelihood (laplace) : {ll_laplace:.6g}")
print(f"AIC (normal) : {aic_norm:.6g}")
print(f"AIC (laplace): {aic_lap:.6g}")
lr = np.exp(ll_normal - ll_laplace)
if lr > 1:
    print(f"Normal favored by likelihood ratio = {lr:.2f}x")
else:
    print(f"Laplace favored by likelihood ratio = {1/lr:.2f}x (inverse)")

if aic_norm < aic_lap:
    print("=> AIC prefers NORMAL (RMSE/MSE are justified).")
    suggested = "RMSE is justified as primary metric (residuals appear Gaussian-like)."
else:
    print("=> AIC prefers LAPLACE (MAE/L1 is justified).")
    suggested = "MAE is justified as primary metric (residuals appear Laplace/heavy-tailed)."
print("")

# 11) Print a suggested one-liner you can paste into your report
print("Suggested report sentence (edit X/Y with numeric values above):")
if aic_norm < aic_lap:
    print("Residual diagnostics and likelihood-based comparison indicate that forecast errors are better described by a Gaussian distribution; therefore RMSE is used as the primary evaluation metric, while MAE is reported as a secondary diagnostic.")
else:
    print("Residual diagnostics and likelihood-based comparison indicate that forecast errors are better described by a Laplace (double-exponential) distribution; therefore MAE is used as the primary evaluation metric, while RMSE is reported as a secondary diagnostic.")
# ============================================================================================


# In[77]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ---- assume `res` exists (flattened residuals: true - pred) ----
# If not, compute res the same way as before:
# preds = inverse_scale_y(preds_scaled, y_scaler)
# trues = inverse_scale_y(trues_scaled, y_scaler)
# resid_flat = (trues - preds).reshape(-1)
# res = resid_flat - np.mean(resid_flat)

# Recompute fit params to be safe
sigma_hat = np.sqrt(np.mean(res**2))   # normal sigma MLE (centered residuals)
b_hat = np.mean(np.abs(res))           # Laplace b MLE (centered residuals)

# x range for plotting fitted pdfs (1st to 99th percentile to avoid extreme tails)
x = np.linspace(np.percentile(res, 1), np.percentile(res, 99), 300)
pdf_norm = stats.norm.pdf(x, loc=0, scale=sigma_hat)
pdf_lap  = stats.laplace.pdf(x, loc=0, scale=b_hat)

# ---------- Residual PDF (histogram + Normal & Laplace PDFs) ----------
fig, ax = plt.subplots(figsize=(7,4))

ax.hist(res, bins=80, density=True, alpha=0.6, label='Residuals', edgecolor='none')
ax.plot(x, pdf_norm, lw=2, label=f'Normal PDF (σ={sigma_hat:.3g})')
ax.plot(x, pdf_lap,  lw=2, linestyle='--', label=f'Laplace PDF (b={b_hat:.3g})')

ax.set_title("Residual distribution (Normal & Laplace fits)")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")

# styling: clean, dashed black grid; remove top/right spines; no tick stubs
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0)
ax.grid(True, linestyle='--', linewidth=1.0, color='black', alpha=0.8)

ax.legend()
plt.tight_layout()
plt.savefig("residual_pdf_with_laplace.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------- QQ-plot vs Normal (separate) ----------
fig, ax = plt.subplots(figsize=(7,4))
stats.probplot(res, dist="norm", plot=ax)
ax.set_title("Normal Q–Q plot of residuals")

# styling: match above
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0)
ax.grid(True, linestyle='--', linewidth=1.0, color='black', alpha=0.8)

plt.tight_layout()
plt.savefig("residual_qq_normal.png", dpi=300, bbox_inches='tight')
plt.show()


# In[76]:


plt.figure(figsize=(7, 4))
ax = plt.gca()

stats.probplot(res, dist="norm", plot=ax)
ax.set_title("Normal Q–Q plot of residuals")

# styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0)
ax.grid(True, linestyle='--', linewidth=1.0, color='black', alpha=0.8)

plt.tight_layout()
plt.savefig("residual_qq_normal.png", dpi=300, bbox_inches='tight')
plt.show()


# # Transformer

# In[19]:


import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds time-position information to embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


# In[20]:


class TransformerForecaster(nn.Module):
    """
    Encoder-only Transformer:
    Input:  (B, T, F)
    Output: (B, H)
    """
    def __init__(
        self,
        n_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        horizon=24
    ):
        super().__init__()
        self.horizon = horizon

        # Project features -> model dimension
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=6000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Forecast head from the final token representation
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        # x: (B, T, F)
        z = self.input_proj(x)          # (B, T, d_model)
        z = self.pos_enc(z)             # (B, T, d_model)
        z = self.encoder(z)             # (B, T, d_model)
        last = z[:, -1, :]              # (B, d_model)
        yhat = self.head(last)          # (B, H)
        return yhat


# In[21]:


import copy
import torch.optim as optim

def fit_model(
    model,
    train_loader,
    val_loader,
    y_scaler,
    epochs=5,
    lr=1e-3,
    weight_decay=1e-5,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_state = None
    best_val_rmse = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, y_scaler)

        history.append({
            "epoch": epoch,
            "train_loss_scaled": train_loss,
            **val_metrics
        })

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss(scaled)={train_loss:.4f} | "
            f"val_RMSE={val_metrics['RMSE']:.4f} | "
            f"val_MAE={val_metrics['MAE']:.4f} | "
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# In[22]:


device = "cuda" if torch.cuda.is_available() else "cpu"

n_features = X_train_w.shape[2]
horizon = Y_train_w.shape[1]

transformer = TransformerForecaster(
    n_features=n_features,
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    dropout=0.25,
    horizon=horizon
)

transformer, tf_history = fit_model(
    model=transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    y_scaler=y_scaler,
    epochs=5,
    lr=1e-3,
    device=device
)

test_metrics_tf = evaluate(
    model=transformer,
    loader=test_loader,
    loss_fn=nn.SmoothL1Loss(),
    device=device,
    y_scaler=y_scaler
)

print("TRANSFORMER TEST METRICS:", test_metrics_tf)


# In[23]:


import matplotlib.pyplot as plt

def plot_history(history, title="Training History"):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss_scaled"] for h in history]
    val_rmse = [h["RMSE"] for h in history]
    val_mae = [h["MAE"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss (scaled)")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (scaled)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, val_rmse, label="Val RMSE")
    axes[1].plot(epochs, val_mae, label="Val MAE")
    axes[1].set_title("Validation Metrics (original units)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()

# Example:
plot_history(history, "LSTM Training History")
plot_history(tf_history, "Transformer Training History")


# In[66]:


@torch.no_grad()
def get_predictions(model, loader, device, y_scaler, max_batches=1):
    model.eval()
    preds_scaled = []
    trues_scaled = []
    n = 0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        Yhat = model(X)

        preds_scaled.append(Yhat.detach().cpu().numpy())
        trues_scaled.append(Y.detach().cpu().numpy())

        n += 1
        if n >= max_batches:
            break

    preds_scaled = np.concatenate(preds_scaled, axis=0)  # (N, H)
    trues_scaled = np.concatenate(trues_scaled, axis=0)  # (N, H)

    preds = inverse_scale_y(preds_scaled, y_scaler)
    trues = inverse_scale_y(trues_scaled, y_scaler)
    return preds, trues


def plot_single_forecast(
    preds,
    trues,
    idx=0,
    title="Forecast vs Ground Truth",
    save_path=None
):
    """
    preds, trues: (N, H) in original units
    idx: which sample window to plot
    """
    y_pred = preds[idx]
    y_true = trues[idx]
    h = len(y_true)
    x = np.arange(1, h + 1)

    plt.figure(figsize=(7, 4))

    plt.plot(
        x, y_true,
        label="Ground Truth",
        linewidth=1.8,
        marker='o'
    )
    plt.plot(
        x, y_pred,
        label="Prediction",
        linewidth=1.8,
        linestyle='--',
        marker='s'
    )

    plt.title(title)
    plt.xlabel("Forecast step (hours ahead)")
    plt.ylabel("Target (original units)")

    # 🔹 dark dashed gridlines (same style as before)
    plt.grid(
        True,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.8
    )

    plt.legend()
    plt.tight_layout()
    

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()    
    


# In[69]:


preds_lstm, trues_lstm = get_predictions(
    model, test_loader, device, y_scaler
)

plot_single_forecast(
    preds_lstm,
    trues_lstm,
    idx=0,
    title="LSTM: 24-hour Forecast",
    save_path="lstm_24h_forecast.png"
)


# In[65]:


# Transformer
preds_tf, trues_tf = get_predictions(
    transformer, test_loader, device, y_scaler
)

plot_single_forecast(
    preds_tf,
    trues_tf,
    idx=0,
    title="Transformer: 24-hour Forecast",
    save_path="transformer_24h_forecast.png"
)


# In[25]:


def horizon_wise_rmse(preds, trues):
    """
    preds, trues: (N, H) original units
    returns rmse_per_h: (H,)
    """
    H = trues.shape[1]
    rmse_h = np.zeros(H)
    for h in range(H):
        rmse_h[h] = np.sqrt(np.mean((trues[:, h] - preds[:, h]) ** 2))
    return rmse_h


def plot_horizon_rmse(preds, trues, title="Horizon-wise RMSE"):
    rmse_h = horizon_wise_rmse(preds, trues)
    x = np.arange(1, len(rmse_h) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, rmse_h, label="RMSE per horizon", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Forecast step (hours ahead)")
    plt.ylabel("RMSE (original units)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example:
plot_horizon_rmse(preds_lstm, trues_lstm, "LSTM Horizon-wise RMSE")
plot_horizon_rmse(preds_tf, trues_tf, "Transformer Horizon-wise RMSE")


# In[57]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- helper to convert history list->DataFrame ---
def hist_to_df(history):
    # history: list of dicts with epoch, train_loss_scaled, loss_scaled, MAE, RMSE, MAPE_%
    return pd.DataFrame(history).set_index('epoch')

# Convert
df_lstm = hist_to_df(history)
df_tf   = hist_to_df(tf_history)

# Simple plotting function: training loss (scaled) and val RMSE (original units)
def plot_convergence(df, title='Model', save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = df.index.values

    ax2 = ax.twinx()

    ax.plot(
        epochs, df['train_loss_scaled'],
        label='train loss (scaled)', linestyle='-', marker='o'
    )

    if 'loss_scaled' in df.columns:
        ax.plot(
            epochs, df['loss_scaled'],
            label='val loss (scaled)', linestyle='--', marker='x'
        )

    if 'RMSE' in df.columns:
        ax2.plot(
            epochs, df['RMSE'],
            label='val RMSE',
            linestyle='-', marker='s'
        )
        ax2.set_ylabel('Val RMSE (original units - kW)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Scaled loss')
    ax.set_title(title)
    ax.set_yticks(np.arange(0, 0.41, 0.02))
    ax.set_xticks(np.arange(0, 7, 0.5))
    
    # unified legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')

    # DARK dashed gridlines
    ax.grid(
        True,
        linestyle='--',
        linewidth=1.0,
        color='black',
        alpha=0.8
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# --- Convergence heuristic ---
# We'll detect the earliest epoch after which val RMSE doesn't improve beyond a small tolerance for k consecutive epochs.
def detect_convergence(df, metric='RMSE', tol=1e-3, patience=5):
    vals = df[metric].values
    best = np.inf
    best_epoch = None
    for i, v in enumerate(vals):
        if v < best - tol:
            best = v
            best_epoch = i+1  # epoch index (1-based)
    # detect plateau: last epoch where improvement > tol within patience window
    for start in range(len(vals)-patience):
        window = vals[start:start+patience]
        if np.all(np.abs(window - window.min()) <= tol):
            return start+1  # 1-based epoch where plateau started
    return best_epoch

print("LSTM convergence epoch (heuristic):", detect_convergence(df_lstm, metric='RMSE', tol=1e-4, patience=4))
print("Transformer convergence epoch (heuristic):", detect_convergence(df_tf, metric='RMSE', tol=1e-4, patience=4))


# In[58]:


plot_convergence(
    df_lstm,
    title='LSTM Convergence',
    save_path='lstm_convergence.png'
)


# In[56]:


plot_convergence(
    df_tf,
    title='Transformer Convergence',
    save_path='transformer_convergence.png'
)


# In[27]:


# === Combined experiment + table generator (single cell) ===
import time, copy, math, numpy as np, torch, random, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "figure.facecolor": "white",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "grid.linestyle": ":", "grid.linewidth": 0.7, "grid.alpha": 0.5,
    "xtick.direction": "in", "ytick.direction": "in"
})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- utilities ----------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmse_np(y, yhat): 
    return float(np.sqrt(np.mean((y-yhat)**2)))

# prediction helper (returns inverse-scaled preds & trues as numpy arrays)
def predict_on_loader(model, loader, y_scaler, inverse_scale_y, device=device):
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                Xb, Yb, _ = batch
            else:
                Xb, Yb = batch
            Xb = Xb.to(device); Yb = Yb.to(device)
            Yhat = model(Xb)
            ph = Yhat.detach().cpu().numpy()
            th = Yb.detach().cpu().numpy()
            preds_list.append(ph); trues_list.append(th)
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    preds_orig = inverse_scale_y(preds, y_scaler)
    trues_orig = inverse_scale_y(trues, y_scaler)
    return preds_orig, trues_orig

# simple single-epoch training helper (used in full training loop)
def train_one_epoch(model, train_loader, optimizer, loss_fn, device=device, grad_clip=5.0):
    model.train()
    losses = []
    for batch in train_loader:
        if len(batch)==3:
            Xb, Yb, _ = batch
        else:
            Xb, Yb = batch
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = loss_fn(out, Yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if len(losses)>0 else 0.0

# replay of the training/eval loop from harness
def train_one_model(model, train_loader, val_loader, y_scaler, inverse_scale_y,
                    optimizer, loss_fn, epochs, device=device,
                    early_stop_patience=6, monitor_metric='rmse'):
    model.to(device)
    history = {'train_loss':[], 'val_rmse':[], 'val_mae':[], 'epoch_time':[]}
    best_val = float('inf'); best_state = None; no_imp=0; best_epoch=0
    for ep in range(1, epochs+1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device=device)
        preds_val, trues_val = predict_on_loader(model, val_loader, y_scaler, inverse_scale_y, device=device)
        val_mae = float(mean_absolute_error(trues_val.reshape(-1), preds_val.reshape(-1)))
        val_rmse = float(rmse_np(trues_val.reshape(-1), preds_val.reshape(-1)))
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['epoch_time'].append(time.time() - t_ep)
        monitor_value = val_rmse if monitor_metric=='rmse' else val_mae
        if monitor_value < best_val - 1e-8:
            best_val = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= early_stop_patience:
            break
    history['best_state'] = best_state
    history['best_epoch'] = best_epoch
    return history

def predict_numpy_with_model(model, X_np, batch_size=128, device=device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.tensor(X_np[i:i+batch_size]).float().to(device)
            ph = model(xb).detach().cpu().numpy()
            preds.append(ph)
    return np.concatenate(preds, axis=0)

# ---------------- fallback model definitions (only used if your classes aren't present) ----------------
# These are minimal models with matching (B,T,F) -> (B,H) behavior to allow the harness to run.
class _FallbackLSTM(torch.nn.Module):
    def __init__(self, n_features=10, hidden_size=64, num_layers=1, horizon=24, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = torch.nn.Linear(hidden_size, horizon)
    def forward(self, x):
        # x: (B, T, F) -> take last hidden output
        out, _ = self.lstm(x)
        last = out[:, -1, :]   # (B, hidden)
        return self.head(last)

class _FallbackTransformer(torch.nn.Module):
    def __init__(self, n_features=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, horizon=24, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(n_features, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = torch.nn.Linear(d_model, horizon)
    def forward(self, x):
        x2 = self.input_proj(x)  # (B,T,d_model)
        enc = self.encoder(x2)    # (B,T,d_model)
        pooled = enc[:, -1, :]    # (B,d_model)
        return self.head(pooled)

# ---------------- builders (use your classes if available, otherwise fallback) ----------------
def build_model_fn_lstm(seed, *, n_features=10, hidden_size=64, num_layers=1, dropout=0.2, horizon=24):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if 'LSTMForecaster' in globals():
        try:
            return LSTMForecaster(n_features=n_features, hidden_size=hidden_size,
                                  num_layers=num_layers, dropout=dropout, horizon=horizon)
        except Exception:
            pass
    # fallback
    return _FallbackLSTM(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers, horizon=horizon, dropout=dropout)

def build_model_fn_transformer(seed, *, n_features=10, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, horizon=24):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if 'TransformerForecaster' in globals():
        try:
            return TransformerForecaster(n_features=n_features, d_model=d_model, nhead=nhead,
                                         num_layers=num_layers, dim_feedforward=dim_feedforward,
                                         dropout=dropout, horizon=horizon)
        except Exception:
            pass
    # fallback
    return _FallbackTransformer(n_features=n_features, d_model=d_model, nhead=nhead,
                                num_layers=num_layers, dim_feedforward=dim_feedforward,
                                horizon=horizon, dropout=dropout)

# ---------------- optimizer wrapper ----------------
def optimizer_fn(model, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# ---------------- experiment runner across seeds (aggregates results) ----------------
def run_experiment_over_seeds(build_model_fn, model_name, seeds,
                              train_loader, val_loader, test_loader,
                              y_scaler, inverse_scale_y,
                              optimizer_fn, loss_fn_callable,
                              epochs=30, device=device, early_stop_patience=6):
    all_histories = []
    best_metrics = []
    times_to_best = []
    param_counts = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = build_model_fn(seed)
        param_counts.append(count_params(model))
        optimizer = optimizer_fn(model)
        loss_fn = loss_fn_callable
        hist = train_one_model(model, train_loader, val_loader, y_scaler, inverse_scale_y,
                               optimizer, loss_fn, epochs, device=device, early_stop_patience=early_stop_patience)
        # load best state and evaluate on test
        if hist['best_state'] is not None:
            model.load_state_dict(hist['best_state'])
        preds_test, trues_test = predict_on_loader(model, test_loader, y_scaler, inverse_scale_y, device=device)
        mae_test = mean_absolute_error(trues_test.reshape(-1), preds_test.reshape(-1))
        rmse_test = rmse_np(trues_test.reshape(-1), preds_test.reshape(-1))
        best_metrics.append({'mae':mae_test, 'rmse':rmse_test, 'preds':preds_test, 'trues':trues_test})
        times_to_best.append(sum(hist['epoch_time'][:max(1, hist['best_epoch'])]))  # wall-clock to best
        all_histories.append(hist)
    maes = [m['mae'] for m in best_metrics]
    rmses = [m['rmse'] for m in best_metrics]
    results = {
        'model_name': model_name,
        'seeds': seeds,
        'param_counts': param_counts,
        'histories': all_histories,
        'test_metrics': {'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
                         'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses)},
        'times_to_best': times_to_best,
        'best_preds_trues': best_metrics
    }
    return results


# In[28]:


# ---------------- run experiments (configure seeds/hyperparams here) ----------------
seeds = [111,222,333]          # change or expand
epochs = 7
loss_fn = torch.nn.SmoothL1Loss()
early_stop_patience = 6

print("Running LSTM experiments...")
lstm_results = run_experiment_over_seeds(lambda seed: build_model_fn_lstm(seed, hidden_size=16, num_layers=1, dropout=0.3),
                                        "LSTM", seeds,
                                        train_loader, val_loader, test_loader,
                                        y_scaler, inverse_scale_y,
                                        optimizer_fn, loss_fn, epochs=epochs, device=device, early_stop_patience=early_stop_patience)

print("Running Transformer experiments...")
trans_results = run_experiment_over_seeds(lambda seed: build_model_fn_transformer(seed, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.25),
                                         "Transformer", seeds,
                                         train_loader, val_loader, test_loader,
                                         y_scaler, inverse_scale_y,
                                         optimizer_fn, loss_fn, epochs=epochs, device=device, early_stop_patience=early_stop_patience)

# ---------------- build summary table ----------------
rows = []
for r in [lstm_results, trans_results]:
    row = {
        'model': r['model_name'],
        'n_params_mean': int(np.mean(r['param_counts'])),
        'n_params_std': int(np.std(r['param_counts'])),
        'MAE_mean': r['test_metrics']['mae_mean'],
        'MAE_std': r['test_metrics']['mae_std'],
        'RMSE_mean': r['test_metrics']['rmse_mean'],
        'RMSE_std': r['test_metrics']['rmse_std'],
        'time_to_best_mean_s': float(np.mean(r['times_to_best']))
    }
    rows.append(row)

df = pd.DataFrame(rows)
print("\n=== Summary table ===\n")
print(df.round(4).to_string(index=False))
csv_name = "model_comparison_summary.csv"
df.to_csv(csv_name, index=False)
print(f"\nSaved summary to {csv_name}")

# ---------------- quick plots: learning curves & per-horizon ----------------
def plot_learning_curves(results_list):
    fig, axes = plt.subplots(1,2, figsize=(10,3.6))
    for res in results_list:
        arrs = [np.array(h['train_loss']) for h in res['histories']]
        max_epochs = max(a.shape[0] for a in arrs)
        stacked = np.array([np.pad(a, (0, max_epochs - a.shape[0]), constant_values=a[-1]) for a in arrs])
        mean_train = stacked.mean(0); std_train = stacked.std(0)
        epochs = np.arange(1, mean_train.size+1)
        axes[0].plot(epochs, mean_train, label=res['model_name'])
        axes[0].fill_between(epochs, mean_train-std_train, mean_train+std_train, alpha=0.15)

        arrs_val = [np.array(h['val_rmse']) for h in res['histories']]
        stacked_val = np.array([np.pad(a, (0, max_epochs - a.shape[0]), constant_values=a[-1]) for a in arrs_val])
        mean_val = stacked_val.mean(0); std_val = stacked_val.std(0)
        axes[1].plot(epochs, mean_val, label=res['model_name'])
        axes[1].fill_between(epochs, mean_val-std_val, mean_val+std_val, alpha=0.15)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss (scaled)"); axes[0].set_title("Training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val RMSE (kW)"); axes[1].set_title("Validation RMSE")
    axes[1].legend(frameon=False)
    axes[0].grid(True); axes[1].grid(True)
    plt.tight_layout(); plt.show()

def plot_per_horizon_from_results(results_list):
    plt.figure(figsize=(7,3.2))
    for res in results_list:
        # take first seed's preds/trues for plotting (or average across seeds if you collect that)
        pt = res['best_preds_trues'][0]
        preds = pt['preds']; trues = pt['trues']
        mae_by_h = np.mean(np.abs(trues - preds), axis=0)
        rmse_by_h = np.sqrt(np.mean((trues - preds)**2, axis=0))
        H = mae_by_h.size
        plt.plot(np.arange(1, H+1), mae_by_h, marker='o', lw=1.4, label=f"{res['model_name']} MAE")
        plt.plot(np.arange(1, H+1), rmse_by_h, marker='x', lw=1.2, linestyle='--', label=f"{res['model_name']} RMSE")
    plt.xlabel("Forecast step"); plt.ylabel("Error (kW)")
    plt.legend(frameon=False); plt.grid(True); plt.tight_layout(); plt.show()

plot_learning_curves([lstm_results])


# End of cell


# In[29]:


plot_per_horizon_from_results([lstm_results, trans_results])


# In[30]:


plot_learning_curves([trans_results])

