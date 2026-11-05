# =============================================================================
# Autograding Tests for Workshop 05: Regression, Lagged Predictors & Combinations
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

load_macro_data = getattr(module, "load_macro_data", None)
build_trend_and_fourier_features = getattr(module, "build_trend_and_fourier_features", None)
compute_aicc = getattr(module, "compute_aicc", None)
create_lagged_matrix = getattr(module, "create_lagged_matrix", None)
verify_causal_alignment = getattr(module, "verify_causal_alignment", None)
run_durbin_watson = getattr(module, "run_durbin_watson", None)
run_breusch_godfrey_test = getattr(module, "run_breusch_godfrey_test", None)
bates_granger_combination = getattr(module, "bates_granger_combination", None)
walk_forward_ensemble_evaluation = getattr(module, "walk_forward_ensemble_evaluation", None)


def test_01_load_macro_data():
    """Test 1: Macroeconomic data ingestion and frequency verification (10 pts)."""
    assert callable(load_macro_data), "Function load_macro_data() not implemented"
    data = load_macro_data()
    assert isinstance(data, dict), "load_macro_data() must return a dictionary"
    
    required_keys = ["cpi", "oil", "rsafs", "unrate"]
    for k in required_keys:
        assert k in data, f"Key '{k}' missing from returned data dictionary"
        s = data[k]
        assert isinstance(s, (pd.Series, pd.DataFrame)), f"data['{k}'] must be a pandas Series or DataFrame"
        assert isinstance(s.index, pd.DatetimeIndex), f"data['{k}'] index must be DatetimeIndex"
        assert s.index.is_monotonic_increasing, f"data['{k}'] index must be monotonically increasing"
        assert len(s) > 100, f"data['{k}'] must contain historical observations (>100 rows)"


def test_02_trend_and_fourier_features():
    """Test 2: Deterministic trend splines, Fourier harmonics, and AICc calculation (15 pts)."""
    assert callable(build_trend_and_fourier_features), "Function build_trend_and_fourier_features() not implemented"
    assert callable(compute_aicc), "Function compute_aicc() not implemented"

    dates = pd.date_range("2010-01-01", periods=120, freq="MS")
    dummy_series = pd.Series(np.linspace(10, 50, 120), index=dates)

    # 1. Linear trend + Fourier K=2
    df_feat = build_trend_and_fourier_features(dummy_series, trend_type="linear", K=2)
    assert isinstance(df_feat, pd.DataFrame), "build_trend_and_fourier_features must return a DataFrame"
    assert "trend" in df_feat.columns, "'trend' column missing from features"
    assert "sin_1" in df_feat.columns and "cos_1" in df_feat.columns, "Fourier order 1 terms missing"
    assert "sin_2" in df_feat.columns and "cos_2" in df_feat.columns, "Fourier order 2 terms missing"
    assert len(df_feat) == 120, "Feature dataframe length must match series length"

    # 2. Piecewise spline with knot
    knot = pd.to_datetime("2015-01-01")
    df_spline = build_trend_and_fourier_features(dummy_series, trend_type="piecewise", K=1, knot_date=knot)
    assert "spline_knot" in df_spline.columns, "'spline_knot' column missing for piecewise trend"
    # Prior to knot date, spline feature must be 0
    assert (df_spline.loc[:"2014-12-01", "spline_knot"] == 0).all(), "Spline knot before tau must be zero"
    assert (df_spline.loc["2015-02-01":, "spline_knot"] > 0).all(), "Spline knot after tau must be positive"

    # 3. AICc formula check
    # T=50, ss_res=100.0, k=4 -> AIC = 50*ln(2) + 2*6 = 34.657 + 12 = 46.657
    # AICc = 46.657 + 2*6*7 / (50 - 4 - 3) = 46.657 + 84 / 43 = 48.6108
    aicc = compute_aicc(ss_res=100.0, n_obs=50, n_params=4)
    expected_aicc = 50.0 * np.log(100.0 / 50.0) + 2 * (4 + 2) + (2 * (4 + 2) * (4 + 3)) / (50 - 4 - 3)
    assert np.isclose(aicc, expected_aicc, rtol=1e-3), f"compute_aicc returned {aicc}, expected {expected_aicc}"


def test_03_create_lagged_matrix_and_audit():
    """Test 3: Causal dynamic lag matrix construction and leak-audit assertion (20 pts)."""
    assert callable(create_lagged_matrix), "Function create_lagged_matrix() not implemented"
    assert callable(verify_causal_alignment), "Function verify_causal_alignment() not implemented"

    dates = pd.date_range("2020-01-01", periods=50, freq="MS")
    y = pd.Series(np.arange(50, dtype=float), index=dates, name="y")
    x = pd.Series(np.arange(50, dtype=float) * 2, index=dates, name="x")

    # Safe lag alignment
    X_lags, y_aligned = create_lagged_matrix(y, x, lags=[1, 2], drop_na=True)
    assert isinstance(X_lags, pd.DataFrame), "X_lags must be a DataFrame"
    assert "x_lag1" in X_lags.columns and "x_lag2" in X_lags.columns, "Expected x_lag1 and x_lag2 columns"
    assert len(X_lags) == 48, "Dropping lags=[1, 2] on 50 periods must leave 48 rows"
    assert len(y_aligned) == 48, "y_aligned must match length of X_lags"

    # Value check: row at index 2 (2020-03-01) must have x_lag1 = x(2020-02-01) = 2.0, x_lag2 = x(2020-01-01) = 0.0
    first_date = X_lags.index[0]
    assert X_lags.loc[first_date, "x_lag1"] == 2.0, "x_lag1 must equal x at t-1"
    assert X_lags.loc[first_date, "x_lag2"] == 0.0, "x_lag2 must equal x at t-2"

    # Verify causal alignment passes on legal lags
    is_safe = verify_causal_alignment(X_lags, y_aligned)
    assert is_safe is True, "verify_causal_alignment must return True on causal lags"

    # Deliberate leakage test: inverted shift must raise AssertionError or ValueError
    X_leaked = pd.DataFrame({"x_future": x.shift(-1)}, index=dates).dropna()
    y_leaked = y.loc[X_leaked.index]
    with pytest.raises((AssertionError, ValueError)):
        verify_causal_alignment(X_leaked, y_leaked)


def test_04_residual_diagnostics_dw_bg():
    """Test 4: Durbin-Watson statistic and Breusch-Godfrey LM test (15 pts)."""
    assert callable(run_durbin_watson), "Function run_durbin_watson() not implemented"
    assert callable(run_breusch_godfrey_test), "Function run_breusch_godfrey_test() not implemented"

    # Case 1: White noise residuals -> DW near 2.0
    np.random.seed(42)
    e_wn = np.random.normal(0, 1, 500)
    dw_wn = run_durbin_watson(e_wn)
    assert np.isclose(dw_wn, 2.0, atol=0.25), f"Durbin-Watson for white noise should be near 2.0, got {dw_wn}"

    # Case 2: Highly persistent random walk residuals -> DW near 0.0
    e_rw = np.cumsum(e_wn)
    dw_rw = run_durbin_watson(e_rw)
    assert dw_rw < 0.5, f"Durbin-Watson for non-stationary residuals must be near 0, got {dw_rw}"

    # Case 3: Breusch-Godfrey test
    X = np.column_stack([np.ones(len(e_wn)), np.arange(len(e_wn))])
    bg_result = run_breusch_godfrey_test(residuals=e_wn, X=X, nlags=4)
    assert isinstance(bg_result, dict), "run_breusch_godfrey_test must return a dict"
    assert "lm_stat" in bg_result and "p_value" in bg_result, "Dict must contain 'lm_stat' and 'p_value'"
    assert bg_result["p_value"] > 0.05, "White noise residuals should fail to reject null of no serial correlation"


def test_05_bates_granger_combination_weights():
    """Test 5: Equal-weight and inverse-variance combination schemes (15 pts)."""
    assert callable(bates_granger_combination), "Function bates_granger_combination() not implemented"

    # Model A: MSE = 4.0 (RMSE = 2.0), forecast = 10.0
    # Model B: MSE = 1.0 (RMSE = 1.0), forecast = 12.0
    forecasts = {"model_a": 10.0, "model_b": 12.0}
    errors = {"model_a": 4.0, "model_b": 1.0}

    # Equal weighting
    comb_eq, weights_eq = bates_granger_combination(forecasts, errors, method="equal")
    assert np.isclose(weights_eq["model_a"], 0.5), "Equal weight on Model A must be 0.5"
    assert np.isclose(weights_eq["model_b"], 0.5), "Equal weight on Model B must be 0.5"
    assert np.isclose(comb_eq, 11.0), "Equal weight combined forecast must be 11.0"

    # Inverse-variance weighting
    # 1/MSE_a = 0.25, 1/MSE_b = 1.0 -> sum = 1.25 -> w_a = 0.2, w_b = 0.8
    comb_inv, weights_inv = bates_granger_combination(forecasts, errors, method="inverse_variance")
    assert np.isclose(weights_inv["model_a"], 0.2), f"Inverse-variance weight for Model A should be 0.2, got {weights_inv['model_a']}"
    assert np.isclose(weights_inv["model_b"], 0.8), f"Inverse-variance weight for Model B should be 0.8, got {weights_inv['model_b']}"
    expected_comb = 0.2 * 10.0 + 0.8 * 12.0  # 2.0 + 9.6 = 11.6
    assert np.isclose(comb_inv, expected_comb), f"Inverse-variance forecast should be {expected_comb}, got {comb_inv}"


def test_06_walk_forward_ensemble_evaluation():
    """Test 6: Walk-forward validation and demonstration of combination puzzle (15 pts)."""
    assert callable(walk_forward_ensemble_evaluation), "Function walk_forward_ensemble_evaluation() not implemented"

    # Create synthetic series
    np.random.seed(123)
    dates = pd.date_range("2000-01-01", periods=150, freq="MS")
    oil_growth = pd.Series(np.random.normal(0, 0.05, 150), index=dates, name="oil")
    # Inflation has persistence + oil shock pass-through
    cpi_growth = pd.Series(np.zeros(150), index=dates, name="cpi")
    for t in range(2, 150):
        cpi_growth.iloc[t] = 0.4 * cpi_growth.iloc[t-1] + 0.3 * oil_growth.iloc[t-1] + np.random.normal(0, 0.02)

    results = walk_forward_ensemble_evaluation(cpi_growth, oil_growth, n_splits=5, train_window=80, test_horizon=1)
    assert isinstance(results, dict), "walk_forward_ensemble_evaluation must return a dictionary of model scores"
    
    expected_models = ["naive", "ar_p", "dynamic_ols", "ensemble_equal", "ensemble_inv_var"]
    for m in expected_models:
        assert m in results, f"Model '{m}' missing from walk-forward evaluation results"
        assert results[m] > 0, f"RMSE for model '{m}' must be strictly positive"

    # Confirm that ensembling produces low error (diversification dividend)
    assert results["ensemble_equal"] <= results["naive"], "Equal ensemble should outperform simple naive baseline"


def test_07_answers_to_practical_questions():
    """Test 7: Verification of student written answers to syllabus practical questions (10 pts)."""
    student_nb_path = Path(__file__).resolve().parent.parent / "student" / "workshop_05.ipynb"
    with open(student_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_text = ""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            full_text += "".join(cell.get("source", [])) + "\n"

    # Practical Question 1: Creating lagged predictors and shift direction trap
    assert ("shift" in full_text.lower() and ("leak" in full_text.lower() or "look-ahead" in full_text.lower())),         "Missing explanation for Q1: shift direction and leakage consequences"

    # Practical Question 2: Choosing combination weights without peeking
    assert ("weight" in full_text.lower() and ("training" in full_text.lower() or "validation" in full_text.lower() or "peeking" in full_text.lower())),         "Missing explanation for Q2: estimating combination weights on training folds without holdout peeking"

    # Practical Question 3: Residual autocorrelation diagnostics before standard errors
    assert ("durbin" in full_text.lower() or "autocorrelation" in full_text.lower() or "standard error" in full_text.lower()),         "Missing explanation for Q3: residual autocorrelation diagnostics and standard error invalidation"
