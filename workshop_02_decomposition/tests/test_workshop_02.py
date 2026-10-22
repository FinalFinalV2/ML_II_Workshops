# =============================================================================
# Autograding Tests for Workshop 02: Time Series Decomposition & Feature Engineering
# Total: 100 Points across 8 Test Cases
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
apply_calendar_and_price_adjustments = getattr(module, "apply_calendar_and_price_adjustments", None)
estimate_guerrero_lambda = getattr(module, "estimate_guerrero_lambda", None)
apply_boxcox = getattr(module, "apply_boxcox", None)
invert_boxcox_mean = getattr(module, "invert_boxcox_mean", None)
compute_centered_ma = getattr(module, "compute_centered_ma", None)
decompose_log_stl = getattr(module, "decompose_log_stl", None)
extract_feasts_features = getattr(module, "extract_feasts_features", None)
audit_table_shift_leakage = getattr(module, "audit_table_shift_leakage", None)
evaluate_imputation_leakage = getattr(module, "evaluate_imputation_leakage", None)
forecast_stl_decomposition = getattr(module, "forecast_stl_decomposition", None)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_calendar_and_price_adjustments():
    """
    [10 Points] Verifies calendar day normalization and price level deflation.
    Checks DatetimeIndex alignment, leap-year days per month, and column arithmetic.
    """
    assert callable(apply_calendar_and_price_adjustments), "apply_calendar_and_price_adjustments not defined"
    sales = load_and_index_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", "MS")
    cpi = load_and_index_series(str(DATA_DIR / "CPIAUCSL.csv"), "observation_date", "CPIAUCSL", "MS")

    df = apply_calendar_and_price_adjustments(sales, cpi)

    assert isinstance(df, pd.DataFrame), "Must return a pd.DataFrame"
    expected_cols = ["raw_sales", "cpi", "real_sales", "days_in_month", "real_sales_per_day"]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    assert df.index.is_monotonic_increasing, "Index must be monotonically increasing"

    # Leap year verification
    # 1992 was a leap year (February has 29 days), 1993 was not (28 days)
    if pd.Timestamp("1992-02-01") in df.index:
        assert df.loc["1992-02-01", "days_in_month"] == 29, "Leap year Feb 1992 must have 29 days"
    if pd.Timestamp("1993-02-01") in df.index:
        assert df.loc["1993-02-01", "days_in_month"] == 28, "Non-leap year Feb 1993 must have 28 days"

    # Verify arithmetic consistency
    base_cpi = float(df["cpi"].iloc[-1])
    expected_real = (df["raw_sales"] / df["cpi"]) * base_cpi
    np.testing.assert_allclose(df["real_sales"].values, expected_real.values, rtol=1e-5)

    expected_daily = df["real_sales"] / df["days_in_month"]
    np.testing.assert_allclose(df["real_sales_per_day"].values, expected_daily.values, rtol=1e-5)


def test_guerrero_and_boxcox_inversion():
    """
    [15 Points] Verifies Guerrero lambda optimization and Box-Cox inversion.
    Checks CV minimization, exact back-transformation, and bias correction logic.
    """
    assert callable(estimate_guerrero_lambda), "estimate_guerrero_lambda not defined"
    assert callable(apply_boxcox), "apply_boxcox not defined"
    assert callable(invert_boxcox_mean), "invert_boxcox_mean not defined"

    sales = load_and_index_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", "MS")
    lmbda = estimate_guerrero_lambda(sales, season_length=12)

    # Retail sales exhibits heteroskedastic seasonal scale; optimal lambda is typically between 0.1 and 0.4
    assert 0.05 <= lmbda <= 0.50, f"Expected retail lambda in [0.05, 0.50], got {lmbda}"

    # Forward and exact inverse roundtrip
    transformed = apply_boxcox(sales, lmbda)
    recovered = invert_boxcox_mean(transformed, lmbda)
    np.testing.assert_allclose(sales.values, recovered.values, rtol=1e-5)

    # Invert for lambda = 0 (log transform)
    log_transformed = apply_boxcox(sales, 0.0)
    recovered_log = invert_boxcox_mean(log_transformed, 0.0)
    np.testing.assert_allclose(sales.values, recovered_log.values, rtol=1e-5)

    # Bias correction check: with sigma2 > 0, back-transformed mean forecast must be strictly greater
    # than uncorrected back-transformed forecast (due to Jensen's inequality)
    mean_uncorrected = invert_boxcox_mean(transformed, lmbda, sigma2=None)
    mean_corrected = invert_boxcox_mean(transformed, lmbda, sigma2=0.04)
    assert (mean_corrected > mean_uncorrected).all(), "Bias-corrected mean must exceed simple inverse for positive series"


def test_centered_ma_endpoint_voids():
    """
    [10 Points] Verifies centered moving average calculation and endpoint void preservation.
    Confirms zero phase distortion and 2 x m MA weight distribution.
    """
    assert callable(compute_centered_ma), "compute_centered_ma not defined"
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    # Linear trend: y_t = 100 + 2 * t
    linear_series = pd.Series(100.0 + 2.0 * np.arange(60), index=idx)

    res = compute_centered_ma(linear_series, period=12)

    assert isinstance(res, pd.DataFrame), "Must return pd.DataFrame"
    for col in ["raw", "centered_trend", "detrended_diff", "detrended_ratio"]:
        assert col in res.columns, f"Missing column {col}"

    trend = res["centered_trend"]
    # For period = 12, exactly 6 points at start and 6 points at end must be NaN
    assert trend.iloc[:6].isna().all(), "First 6 observations must be NaN"
    assert trend.iloc[-6:].isna().all(), "Last 6 observations must be NaN"
    assert not trend.iloc[6:-6].isna().any(), "Interior observations must not be NaN"

    # Symmetric centered MA on linear series reproduces linear trend exactly
    interior_actual = trend.iloc[6:-6].values
    interior_expected = linear_series.iloc[6:-6].values
    np.testing.assert_allclose(interior_actual, interior_expected, rtol=1e-5)


def test_log_stl_multiplicative_reconstruction():
    """
    [15 Points] Direction 1: Verifies Multiplicative Log-STL decomposition.
    Ensures exact multiplicative reconstruction Y_t = T_t * S_t * R_t at machine precision.
    """
    assert callable(decompose_log_stl), "decompose_log_stl not defined"
    sales = load_and_index_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", "MS")

    decomp = decompose_log_stl(sales, period=12, seasonal=13, robust=True)

    assert isinstance(decomp, pd.DataFrame), "Must return pd.DataFrame"
    for col in ["observed", "trend", "seasonal", "remainder", "seasonally_adjusted"]:
        assert col in decomp.columns, f"Missing decomposition component {col}"

    # Direction 1: Exact multiplicative reconstruction
    reconstructed = decomp["trend"] * decomp["seasonal"] * decomp["remainder"]
    max_relative_err = np.max(np.abs((sales.values - reconstructed.values) / sales.values))
    assert max_relative_err < 1e-10, (
        f"Multiplicative reconstruction violated: max relative error {max_relative_err:.2e} >= 1e-10"
    )

    # Seasonally adjusted series identity: A_t = Y_t / S_t
    np.testing.assert_allclose(
        decomp["seasonally_adjusted"].values,
        (decomp["observed"] / decomp["seasonal"]).values,
        rtol=1e-5
    )

    # Should raise error on non-positive series
    neg_series = pd.Series([-10.0, 5.0, 15.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    with pytest.raises(ValueError):
        decompose_log_stl(neg_series)


def test_feasts_feature_extraction():
    """
    [15 Points] Verifies FEASTS summary feature extraction:
    F_T trend strength, F_S seasonal strength, peak/trough month, and remainder diagnostics.
    """
    assert callable(extract_feasts_features), "extract_feasts_features not defined"
    sales = load_and_index_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", "MS")
    decomp = decompose_log_stl(sales, period=12, seasonal=13, robust=True)

    features = extract_feasts_features(decomp, is_multiplicative=True)

    required_keys = [
        "trend_strength", "seasonal_strength", "peak_month", "trough_month",
        "remainder_acf1", "remainder_skew", "remainder_kurtosis"
    ]
    for k in required_keys:
        assert k in features, f"Missing feature key: {k}"

    # Bounds on variance ratios
    assert 0.0 <= features["trend_strength"] <= 1.0, "Trend strength F_T must be in [0, 1]"
    assert 0.0 <= features["seasonal_strength"] <= 1.0, "Seasonal strength F_S must be in [0, 1]"

    # US Retail Sales has massive macro trend and massive holiday seasonality
    assert features["trend_strength"] > 0.85, f"Expected high trend strength, got {features['trend_strength']}"
    assert features["seasonal_strength"] > 0.70, f"Expected high seasonal strength, got {features['seasonal_strength']}"

    # December is the peak retail month in the US (Christmas shopping)
    assert features["peak_month"] == 12, f"Expected retail peak in December (month 12), got month {features['peak_month']}"


def test_table_shift_leakage_and_causal_imputation():
    """
    [15 Points] Verifies temporal hygiene auditing and causal vs leaked imputation.
    """
    assert callable(audit_table_shift_leakage), "audit_table_shift_leakage not defined"
    assert callable(evaluate_imputation_leakage), "evaluate_imputation_leakage not defined"

    # Test 1: Clean temporal split
    idx_train = pd.date_range("2010-01-01", "2019-12-01", freq="MS")
    idx_test = pd.date_range("2020-01-01", "2023-12-01", freq="MS")
    train = pd.Series(np.random.randn(len(idx_train)), index=idx_train)
    test = pd.Series(np.random.randn(len(idx_test)), index=idx_test)

    report = audit_table_shift_leakage(train, test)
    assert report["has_overlap"] is False
    assert report["is_strictly_causal"] is True

    # Test 2: Overlapping split (data leakage)
    idx_overlap_test = pd.date_range("2019-06-01", "2022-12-01", freq="MS")
    test_leaked = pd.Series(np.random.randn(len(idx_overlap_test)), index=idx_overlap_test)
    leaked_report = audit_table_shift_leakage(train, test_leaked)
    assert leaked_report["has_overlap"] is True
    assert leaked_report["is_strictly_causal"] is False

    # Test 3: Imputation leakage
    idx_nan = pd.date_range("2020-01-01", periods=6, freq="MS")
    s_nan = pd.Series([10.0, np.nan, np.nan, 40.0, 50.0, 60.0], index=idx_nan)
    imp = evaluate_imputation_leakage(s_nan)

    assert "bfill_leaked" in imp and "causal_filled" in imp
    # At index 1, bfill/interpolation uses observation 40.0 from period 3 (anticipating future)
    assert imp["bfill_leaked"].iloc[1] > 10.0, "Leaked interpolation should use future values"
    # Causal forward fill strictly carries forward 10.0
    assert imp["causal_filled"].iloc[1] == 10.0, "Causal forward fill must never use future data"


def test_stl_decomposition_forecaster():
    """
    [10 Points] Direction 4: Verifies Production STLF Forecaster.
    Checks drift on seasonally adjusted series, seasonal naive projection, and recombination.
    """
    assert callable(forecast_stl_decomposition), "forecast_stl_decomposition not defined"
    sales = load_and_index_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", "MS")

    horizon = 12
    fc_results = forecast_stl_decomposition(sales, horizon=horizon, period=12, robust=True)

    required_keys = ["forecast", "trend_forecast", "seasonal_forecast", "in_sample_fitted"]
    for k in required_keys:
        assert k in fc_results, f"Missing key in forecaster output: {k}"

    fc = fc_results["forecast"]
    assert isinstance(fc, pd.Series), "Forecast must be a pd.Series"
    assert len(fc) == horizon, f"Forecast length must equal horizon={horizon}, got {len(fc)}"
    assert (fc > 0).all(), "All forecasted sales values must be positive"

    # Future index continuity check
    expected_start = sales.index[-1] + pd.DateOffset(months=1)
    assert fc.index[0] == expected_start, f"Forecast must start at {expected_start}, got {fc.index[0]}"

    # Multiplicative recombination verification: Y_hat = A_hat * S_hat
    recombined = fc_results["trend_forecast"] * fc_results["seasonal_forecast"]
    np.testing.assert_allclose(fc.values, recombined.values, rtol=1e-5)


def test_interview_knowledge_questions():
    """
    [10 Points] Verifies that the 3 syllabus interview questions are answered substantive in markdown.
    """
    target = os.environ.get("AUTOGRADE_TARGET", "auto").lower()
    if target in ("instructor", "solution"):
        assert True
        return

    student_nb_path = Path(__file__).resolve().parent.parent / "student" / "workshop_02.ipynb"
    if not student_nb_path.exists():
        pytest.skip("Student notebook not found for interview questions check")

    with open(student_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    questions_found = 0
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if any(q in src for q in ["### Question 1", "### Question 2", "### Question 3"]):
            if i + 1 < len(cells):
                ans_cell = cells[i + 1]
                ans_src = "".join(ans_cell.get("source", [])).strip()
                if len(ans_src) > 30 and "*Write your answer here in markdown...*" not in ans_src:
                    questions_found += 1

    assert questions_found >= 3, (
        f"Please answer all 3 interview questions in markdown cells (found {questions_found}/3 substantive answers)."
    )
