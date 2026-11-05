#!/usr/bin/env python3
"""
Timestral-28M Hands-on Sandbox & Benchmarking Suite
Comparing Mistral's Zero-Shot Time Series Foundation Model against
Workshop 02 Classical Decompositions on Real-World Macroeconomic Data.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from timestral import TimestralForecaster
from statsmodels.tsa.seasonal import STL

# Set plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 130

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent / "data"

print("=" * 70)
print("  TIMESTRAL-28M ZERO-SHOT FORECASTING PLAYGROUND (MISTRAL AI)")
print("=" * 70)

# 1. Initialize Model
print("\n[1/4] Loading Timestral-28M on Apple Silicon (CPU)...")
t0 = time.perf_counter()
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-28M", device="cpu")
load_time = time.perf_counter() - t0
print(f"  -> Model ready in {load_time:.2f}s!")

# =============================================================================
# EXPERIMENT 1: Synthetic Trend & Multi-Period Harmonic Wave
# =============================================================================
print("\n[2/4] Running Experiment 1: Synthetic Multi-Frequency Extrapolation...")
np.random.seed(42)
N_TRAIN = 180
N_TEST = 24
total_steps = N_TRAIN + N_TEST
time_axis = np.arange(total_steps)

# Signal: Linear trend + 12-step cycle (annual) + 4-step cycle (quarterly) + noise
trend = 0.04 * time_axis + 10.0
seasonal_12 = 2.5 * np.sin(2 * np.pi * time_axis / 12)
seasonal_4 = 1.0 * np.cos(2 * np.pi * time_axis / 4)
noise = np.random.normal(0, 0.35, size=total_steps)
y_all = trend + seasonal_12 + seasonal_4 + noise

y_train = y_all[:N_TRAIN]
y_test = y_all[N_TRAIN:]

t0 = time.perf_counter()
fc_synth = forecaster.forecast(y_train, prediction_length=N_TEST)
infer_synth_ms = (time.perf_counter() - t0) * 1000

synth_median = fc_synth.median.squeeze().numpy()
synth_q10 = fc_synth.quantile(0.10).squeeze().numpy()
synth_q90 = fc_synth.quantile(0.90).squeeze().numpy()

mae_synth = np.mean(np.abs(y_test - synth_median))
print(f"  -> Forward pass completed in {infer_synth_ms:.1f}ms | Out-of-sample MAE: {mae_synth:.4f}")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(time_axis[:N_TRAIN], y_train, color="#1F77B4", label="Historical Context ($L=180$)", lw=1.5)
ax.plot(time_axis[N_TRAIN:], y_test, color="black", ls="--", label="Ground Truth Future ($H=24$)", lw=1.5)
ax.plot(time_axis[N_TRAIN:], synth_median, color="#D62728", label="Timestral-28M Median ($\hat{Y}_{50}$)", lw=2, marker='o', markersize=3)
ax.fill_between(time_axis[N_TRAIN:], synth_q10, synth_q90, color="#D62728", alpha=0.2, label="80% Uncertainty Band ($q_{10}-q_{90}$)")
ax.axvline(N_TRAIN, color="gray", ls=":", alpha=0.7)
ax.set_title("Experiment 1: Zero-Shot Multi-Frequency Extrapolation (Synthetic)", fontsize=12, fontweight='bold')
ax.set_xlabel("Time Steps")
ax.set_ylabel("Signal Amplitude")
ax.legend(loc="upper left")
plt.tight_layout()
exp1_path = OUT_DIR / "images" / "exp1_synthetic_extrapolation.png"
plt.savefig(exp1_path)
plt.close()
print(f"  -> Saved plot to {exp1_path.name}")

# =============================================================================
# EXPERIMENT 2 & 3: The FRED Real-World Showdown on US Retail Sales (RSAFS)
# =============================================================================
print("\n[3/4] Running Experiment 2 & 3: Real Macroeconomic Showdown (US Retail Sales)...")
rsafs_df = pd.read_csv(DATA_DIR / "RSAFS.csv")
rsafs_df['observation_date'] = pd.to_datetime(rsafs_df['observation_date'])
sales = rsafs_df.set_index('observation_date')['RSAFSNA'].astype(float).asfreq('MS')

# Train on data up to August 2025; Test on the final 12 months (Sep 2025 - Aug 2026)
H = 12
train_sales = sales.iloc[:-H]
test_sales = sales.iloc[-H:]

# Method 1: Timestral-28M Zero-Shot
t0 = time.perf_counter()
fc_timestral = forecaster.forecast(train_sales.values, prediction_length=H)
timestral_time_ms = (time.perf_counter() - t0) * 1000

timestral_median = fc_timestral.median.squeeze().numpy()
timestral_q05 = fc_timestral.quantile(0.05).squeeze().numpy()
timestral_q25 = fc_timestral.quantile(0.25).squeeze().numpy()
timestral_q75 = fc_timestral.quantile(0.75).squeeze().numpy()
timestral_q95 = fc_timestral.quantile(0.95).squeeze().numpy()

# Method 2: Workshop 02 Multiplicative Log-STL + Drift Forecaster
t0 = time.perf_counter()
log_s = np.log(train_sales)
stl_res = STL(log_s, period=12, seasonal=13, robust=True).fit()
T_comp = np.exp(stl_res.trend)
S_comp = np.exp(stl_res.seasonal)
R_comp = np.exp(stl_res.resid)
A_comp = T_comp * R_comp

# Drift on seasonally adjusted series
A_T = float(A_comp.iloc[-1])
A_1 = float(A_comp.iloc[0])
slope = (A_T - A_1) / (len(A_comp) - 1)
A_future = A_T + np.arange(1, H + 1) * slope
S_future = S_comp.iloc[-12:].values
stlf_forecast = A_future * S_future
stlf_time_ms = (time.perf_counter() - t0) * 1000

# Method 3: Seasonal Naive Baseline (value from 12 months prior)
t0 = time.perf_counter()
snaive_forecast = train_sales.iloc[-H:].values
snaive_time_ms = (time.perf_counter() - t0) * 1000

# Calculate Metrics
def calc_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape

mae_time, rmse_time, mape_time = calc_metrics(test_sales.values, timestral_median)
mae_stlf, rmse_stlf, mape_stlf = calc_metrics(test_sales.values, stlf_forecast)
mae_snaive, rmse_snaive, mape_snaive = calc_metrics(test_sales.values, snaive_forecast)

summary_table = pd.DataFrame({
    "Model Family": [
        "Mistral Timestral-28M (Zero-Shot)",
        "Log-STL + Drift (Workshop 02)",
        "Seasonal Naïve (Classical Baseline)"
    ],
    "MAE ($)": [f"${mae_time:,.0f}", f"${mae_stlf:,.0f}", f"${mae_snaive:,.0f}"],
    "RMSE ($)": [f"${rmse_time:,.0f}", f"${rmse_stlf:,.0f}", f"${rmse_snaive:,.0f}"],
    "MAPE (%)": [f"{mape_time:.2f}%", f"{mape_stlf:.2f}%", f"{mape_snaive:.2f}%"],
    "Inference Time": [f"{timestral_time_ms:.1f} ms", f"{stlf_time_ms:.1f} ms", f"{snaive_time_ms:.3f} ms"],
    "Parameters Tuned": ["0 (Pre-trained)", "0 (Analytical)", "0 (Heuristic)"]
})

print("\n" + "=" * 70)
print("  SHOWDOWN RESULTS: US RETAIL SALES OUT-OF-SAMPLE (H=12 MONTHS)")
print("=" * 70)
print(summary_table.to_string(index=False))

# Plot Real World Showdown
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel 1: Full History with Forecast
recent_idx = sales.iloc[-60:].index
ax1.plot(recent_idx, sales.iloc[-60:] / 1e3, color="#7F7F7F", label="Recent History Actuals", lw=1.2)
ax1.plot(test_sales.index, test_sales / 1e3, color="black", lw=2, label="Ground Truth (Sep 2025 - Aug 2026)")
ax1.plot(test_sales.index, timestral_median / 1e3, color="#D62728", lw=2, marker='o', markersize=4, label="Timestral-28M Median")
ax1.plot(test_sales.index, stlf_forecast / 1e3, color="#2CA02C", ls="--", lw=1.8, label="Workshop 02 Log-STL Drift")
ax1.fill_between(test_sales.index, timestral_q05 / 1e3, timestral_q95 / 1e3, color="#D62728", alpha=0.15, label="Timestral 90% Fan")
ax1.set_ylabel("Sales ($ Billions)")
ax1.set_title("US Retail Sales: Zero-Shot Foundation Model vs Classical Decomposition", fontsize=12, fontweight='bold')
ax1.legend(loc="upper left")

# Panel 2: Error Profile (Actual - Forecast)
err_time = test_sales.values - timestral_median
err_stlf = test_sales.values - stlf_forecast
ax2.plot(test_sales.index, err_time / 1e3, color="#D62728", marker='s', lw=1.5, label=f"Timestral-28M Residual (MAE: ${mae_time:,.0f})")
ax2.plot(test_sales.index, err_stlf / 1e3, color="#2CA02C", marker='^', ls="--", lw=1.5, label=f"Log-STL Residual (MAE: ${mae_stlf:,.0f})")
ax2.axhline(0, color="black", ls=":", lw=0.8)
ax2.set_ylabel("Error ($ Billions)")
ax2.set_title("Forecast Error Trajectory Across 12-Month Out-of-Sample Window", fontsize=12, fontweight='bold')
ax2.legend(loc="lower left")

plt.tight_layout()
exp2_path = OUT_DIR / "images" / "exp2_retail_sales_showdown.png"
plt.savefig(exp2_path)
plt.close()
print(f"  -> Saved plot to {exp2_path.name}")

# =============================================================================
# EXPERIMENT 4: Structural Break & Attention Recovery (COVID Shock)
# =============================================================================
print("\n[4/4] Running Experiment 4: The COVID Structural Break Test...")
# Cut history at Feb 2020 (right before lockdown collapse in Mar/Apr 2020)
pre_covid = sales.loc[:"2020-02-01"]
covid_actual = sales.loc["2020-03-01":"2021-02-01"]  # 12 months of shock

fc_covid = forecaster.forecast(pre_covid.values, prediction_length=12)
covid_median = fc_covid.median.squeeze().numpy()
covid_q10 = fc_covid.quantile(0.10).squeeze().numpy()
covid_q90 = fc_covid.quantile(0.90).squeeze().numpy()

fig, ax = plt.subplots(figsize=(11, 4.5))
view_idx = sales.loc["2018-01-01":"2021-06-01"].index
ax.plot(view_idx, sales.loc[view_idx] / 1e3, color="black", label="Actual History & Crash (COVID)", lw=1.5)
ax.plot(covid_actual.index, covid_median / 1e3, color="#D62728", lw=2, marker='o', markersize=3, label="Timestral-28M Counterfactual (No Shock Known)")
ax.fill_between(covid_actual.index, covid_q10 / 1e3, covid_q90 / 1e3, color="#D62728", alpha=0.2, label="80% Uncertainty Band")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), color="red", alpha=0.15, label="Exogenous Lockdown Shock")
ax.set_ylabel("Sales ($ Billions)")
ax.set_title("Experiment 4: Zero-Shot Behavior at a Black Swan Event (March 2020 Lockdown)", fontsize=12, fontweight='bold')
ax.legend(loc="upper left")
plt.tight_layout()
exp4_path = OUT_DIR / "images" / "exp4_covid_structural_break.png"
plt.savefig(exp4_path)
plt.close()
print(f"  -> Saved plot to {exp4_path.name}")

print("\n" + "=" * 70)
print("  EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
