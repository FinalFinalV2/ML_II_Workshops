# =============================================================================
# Autograding Tests for Workshop 03: The Forecaster\x27s Toolbox (Benchmarks & Diagnostics)
# Total: 100 Points across 7 Test Cases
# =============================================================================

import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from nb_loader import load_submission

module = load_submission()

load_and_index_series = getattr(module, "load_and_index_series", None)
forecast_mean = getattr(module, "forecast_mean", None)
forecast_naive = getattr(module, "forecast_naive", None)
forecast_snaive = getattr(module, "forecast_snaive", None)
forecast_drift = getattr(module, "forecast_drift", None)
compute_benchmark_residuals = getattr(module, "compute_benchmark_residuals", None)
compute_forecast_sd = getattr(module, "compute_forecast_sd", None)
compute_prediction_intervals = getattr(module, "compute_prediction_intervals", None)
bootstrap_prediction_intervals = getattr(module, "bootstrap_prediction_intervals", None)
check_residuals = getattr(module, "check_residuals", None)
evaluate_empirical_coverage = getattr(module, "evaluate_empirical_coverage", None)
forecast_stlf = getattr(module, "forecast_stlf", None)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_and_index_series():
    """
    [10 Points] Verifies ingestion and enforcement of DatetimeIndex and frequency.
    """
    assert callable(load_and_index_series), "load_and_index_series function missing"
    beer = load_and_index_series(str(DATA_DIR / "aus_beer.csv"), "observation_date", "beer_production", freq="QS")
    assert isinstance(beer, pd.Series), "Must return a pd.Series"
    assert isinstance(beer.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    assert beer.index.is_monotonic_increasing, "Index must be monotonically increasing"
    assert beer.name == "beer_production", "Series name must be preserved"
    assert len(beer) > 10, "Series must contain data"


def test_benchmark_forecasting_engines():
    """
    [20 Points] Verifies point forecasting logic for Mean, Naive, SNaive, and Drift.
    """
    assert callable(forecast_mean), "forecast_mean missing"
    assert callable(forecast_naive), "forecast_naive missing"
    assert callable(forecast_snaive), "forecast_snaive missing"
    assert callable(forecast_drift), "forecast_drift missing"

    # Synthetic monthly test series: [10, 20, 30, 40]
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    y = pd.Series([10.0, 20.0, 30.0, 40.0], index=dates, name="test_target")

    h = 3
    # 1. Mean forecast: (10+20+30+40)/4 = 25.0
    fc_m = forecast_mean(y, h=h)
    assert len(fc_m) == h, "Mean forecast length mismatch"
    assert np.allclose(fc_m.values, [25.0, 25.0, 25.0]), "Mean forecast calculation incorrect"
    assert fc_m.index[0] == pd.Timestamp("2020-05-01"), "Future index must start after train end"

    # 2. Naive forecast: y_T = 40.0
    fc_n = forecast_naive(y, h=h)
    assert len(fc_n) == h, "Naive forecast length mismatch"
    assert np.allclose(fc_n.values, [40.0, 40.0, 40.0]), "Naive forecast calculation incorrect"

    # 3. Seasonal Naive: with m=2, should repeat [30, 40, 30]
    fc_sn = forecast_snaive(y, h=h, m=2)
    assert len(fc_sn) == h, "SNaive forecast length mismatch"
    assert np.allclose(fc_sn.values, [30.0, 40.0, 30.0]), "Seasonal Naive modulo calculation incorrect"

    # 4. Drift forecast: slope = (40 - 10)/(4 - 1) = 10.0 => 40 + 10 = 50, 60, 70
    fc_d = forecast_drift(y, h=h)
    assert len(fc_d) == h, "Drift forecast length mismatch"
    assert np.allclose(fc_d.values, [50.0, 60.0, 70.0]), "Drift slope calculation incorrect"


def test_benchmark_residuals():
    """
    [15 Points] Verifies one-step in-sample residuals and missing-value NaN geometry.
    """
    assert callable(compute_benchmark_residuals), "compute_benchmark_residuals missing"
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    y = pd.Series([10.0, 15.0, 25.0, 30.0, 35.0, 45.0], index=dates, name="val")

    # Mean: 0 NaNs
    res_m = compute_benchmark_residuals(y, method="mean")
    assert res_m.isna().sum() == 0, "Mean method residuals should have 0 NaNs"
    assert np.isclose(res_m.mean(), 0.0), "Mean residuals must sum to zero"

    # Naive: 1 NaN at index 0
    res_n = compute_benchmark_residuals(y, method="naive")
    assert res_n.isna().sum() == 1, "Naive method residuals must have exactly 1 NaN at t=1"
    assert np.isnan(res_n.iloc[0]), "First residual must be NaN"
    assert np.isclose(res_n.iloc[1], 15.0 - 10.0), "One-step error e_2 = y_2 - y_1"

    # SNaive: m NaNs at start
    res_sn = compute_benchmark_residuals(y, method="snaive", m=2)
    assert res_sn.isna().sum() == 2, "SNaive method residuals must have exactly m NaNs at start"
    assert np.isclose(res_sn.iloc[2], 25.0 - 10.0), "e_3 = y_3 - y_1 with m=2"

    # Drift: 1 NaN at start
    res_d = compute_benchmark_residuals(y, method="drift")
    assert res_d.isna().sum() == 1, "Drift residuals must have exactly 1 NaN at start"


def test_forecast_standard_deviations_table52():
    """
    [15 Points] Verifies Table 5.2 multi-step standard error formulas and scaling.
    """
    assert callable(compute_forecast_sd), "compute_forecast_sd missing"
    # Constant error series e_t = 2.0 (length 10)
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    res = pd.Series([2.0] * 10, index=dates)

    # 1. Naive: sigma * sqrt(h)
    sd_n1 = compute_forecast_sd(res, method="naive", h=1, T=10)
    sd_n4 = compute_forecast_sd(res, method="naive", h=4, T=10)
    assert np.isclose(sd_n4, 2.0 * sd_n1), "Naive SD must scale exactly as sqrt(h): sd(4) = 2 * sd(1)"

    # 2. SNaive: sigma * sqrt(k + 1) where k = floor((h-1)/m)
    # For m=4, h=1..4 has k=0 => sqrt(1); h=5 has k=1 => sqrt(2)
    sd_sn1 = compute_forecast_sd(res, method="snaive", h=1, T=10, m=4)
    sd_sn4 = compute_forecast_sd(res, method="snaive", h=4, T=10, m=4)
    sd_sn5 = compute_forecast_sd(res, method="snaive", h=5, T=10, m=4)
    assert np.isclose(sd_sn1, sd_sn4), "SNaive SD must remain constant within same year (h=1..4)"
    assert np.isclose(sd_sn5, sd_sn1 * np.sqrt(2.0)), "SNaive SD must jump by sqrt(2) at h=5 (new seasonal cycle)"

    # 3. Drift: sigma * sqrt(h * (1 + h/(T-1)))
    sd_d1 = compute_forecast_sd(res, method="drift", h=1, T=10)
    expected_d1_factor = np.sqrt(1.0 * (1.0 + 1.0 / 9.0))
    # Note: K=1 for drift, valid=10 => sigma_hat = sqrt(40 / 9) = 2.108
    sigma_hat = np.sqrt(10 * 4.0 / (10 - 1))
    assert np.isclose(sd_d1, sigma_hat * expected_d1_factor), "Drift Table 5.2 formula mismatch"


def test_prediction_intervals_and_bootstrapping():
    """
    [15 Points] Verifies analytical and bootstrapped prediction intervals.
    """
    assert callable(compute_prediction_intervals), "compute_prediction_intervals missing"
    assert callable(bootstrap_prediction_intervals), "bootstrap_prediction_intervals missing"

    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    np.random.seed(42)
    y = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 30)), index=dates, name="stock")

    h = 4
    # Analytical intervals
    fc = forecast_naive(y, h=h)
    res = compute_benchmark_residuals(y, method="naive")
    df_pi = compute_prediction_intervals(fc, res, method="naive", levels=[80, 95])

    expected_cols = ["point_forecast", "lower_80", "upper_80", "lower_95", "upper_95"]
    for col in expected_cols:
        assert col in df_pi.columns, f"Missing interval column: {col}"

    # Monotonic interval width: 95% band must be strictly wider than 80% band
    assert (df_pi["lower_95"] < df_pi["lower_80"]).all(), "Lower 95% must be below lower 80%"
    assert (df_pi["upper_95"] > df_pi["upper_80"]).all(), "Upper 95% must be above upper 80%"

    # Expanding width with horizon h: interval width at h=4 must be wider than h=1
    width_h1 = df_pi["upper_95"].iloc[0] - df_pi["lower_95"].iloc[0]
    width_h4 = df_pi["upper_95"].iloc[3] - df_pi["lower_95"].iloc[3]
    assert width_h4 > width_h1, "Prediction intervals must expand with horizon h"

    # Bootstrapped intervals
    df_boot = bootstrap_prediction_intervals(y, method="naive", h=h, B=500, levels=[80, 95], random_state=42)
    for col in expected_cols:
        assert col in df_boot.columns, f"Missing boot column: {col}"
    assert (df_boot["lower_95"] < df_boot["upper_95"]).all(), "Bootstrapped lower bound must be < upper"


def test_check_residuals_diagnostics():
    """
    [15 Points] Verifies automated diagnostic battery: unbiasedness, Ljung-Box, and normality.
    """
    assert callable(check_residuals), "check_residuals missing"

    # 1. Pure standard Gaussian white noise
    np.random.seed(42)
    wn = pd.Series(np.random.normal(0, 1, 200))
    diag_wn = check_residuals(wn, lags=10, K=0)

    required_keys = ["mean", "is_unbiased", "ljung_box_stat", "ljung_box_pvalue", "is_white_noise", "is_normal"]
    for k in required_keys:
        assert k in diag_wn, f"Missing diagnostic key: {k}"

    assert diag_wn["is_unbiased"] is True, "Gaussian white noise must be unbiased"
    assert diag_wn["is_white_noise"] is True, "Gaussian white noise must pass Ljung-Box test"
    assert diag_wn["is_normal"] is True, "Gaussian white noise must pass normality test"

    # 2. Biased autocorrelated series
    biased = pd.Series(np.arange(100) * 0.5 + np.random.normal(5, 1, 100))
    diag_bad = check_residuals(biased, lags=10, K=0)
    assert diag_bad["is_unbiased"] is False, "Biased series must fail unbiasedness test"
    assert diag_bad["is_white_noise"] is False, "Trending autocorrelated series must fail Ljung-Box test"


def test_forecast_stlf():
    """
    [10 Points] Verifies STLF decomposition forecasting workflow on seasonal data.
    """
    assert callable(forecast_stlf), "forecast_stlf missing"
    beer = load_and_index_series(str(DATA_DIR / "aus_beer.csv"), "observation_date", "beer_production", freq="QS")
    train = beer.iloc[:60]
    h = 8
    fc = forecast_stlf(train, h=h, m=4, method_trend="drift")

    assert isinstance(fc, pd.Series), "STLF must return pd.Series"
    assert len(fc) == h, "STLF forecast horizon length mismatch"
    assert fc.index[0] == train.index[-1] + pd.DateOffset(months=3), "Index must align with quarterly calendar"
    # Verify that forecast preserves seasonal variation (not a flat horizontal line)
    assert np.std(fc.values) > 5.0, "STLF forecast must preserve seasonal oscillation"
