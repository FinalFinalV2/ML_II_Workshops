# =============================================================================
# Autograding Tests for Workshop 04: Evaluating Forecasts & Walk-Forward Validation
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

load_macro_series = getattr(module, "load_macro_series", None)
compute_mae = getattr(module, "compute_mae", None)
compute_rmse = getattr(module, "compute_rmse", None)
compute_mape = getattr(module, "compute_mape", None)
compute_mase = getattr(module, "compute_mase", None)
compute_pinball_loss = getattr(module, "compute_pinball_loss", None)
compute_winkler_score = getattr(module, "compute_winkler_score", None)
generate_walk_forward_splits = getattr(module, "generate_walk_forward_splits", None)
demonstrate_kfold_leakage = getattr(module, "demonstrate_kfold_leakage", None)
evaluate_horizon_degradation = getattr(module, "evaluate_horizon_degradation", None)
diebold_mariano_test = getattr(module, "diebold_mariano_test", None)
run_leakage_audit_suite = getattr(module, "run_leakage_audit_suite", None)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_macro_series():
    """
    [10 Points] Verifies ingestion and enforcement of DatetimeIndex and monthly frequency.
    """
    assert callable(load_macro_series), "load_macro_series function missing"
    series = load_macro_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", freq="MS")
    assert isinstance(series, pd.Series), "Must return a pd.Series"
    assert isinstance(series.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    assert series.index.is_monotonic_increasing, "Index must be monotonically increasing"
    assert series.name == "RSAFSNA", "Series name must be preserved"
    assert len(series) >= 300, "Series must contain the full history of RSAFS"


def test_point_accuracy_metrics():
    """
    [15 Points] Verifies numerical accuracy of MAE, RMSE, and MAPE with zero-division guard.
    """
    assert callable(compute_mae), "compute_mae missing"
    assert callable(compute_rmse), "compute_rmse missing"
    assert callable(compute_mape), "compute_mape missing"

    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 16.0, 35.0])

    # Absolute deviations: [2.0, 4.0, 5.0] -> mean = 11/3
    mae = compute_mae(y_true, y_pred)
    assert np.isclose(mae, 11.0 / 3.0), f"Expected MAE 3.667, got {mae}"

    # Squared deviations: [4.0, 16.0, 25.0] -> mean = 45/3 = 15.0 -> sqrt(15) = 3.87298
    rmse = compute_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(15.0)), f"Expected RMSE sqrt(15), got {rmse}"

    # Percentage errors: [2/10 = 20%, 4/20 = 20%, 5/30 = 16.667%] -> mean = 18.889%
    mape = compute_mape(y_true, y_pred)
    assert np.isclose(mape, (20.0 + 20.0 + 50.0 / 3.0) / 3.0), f"Expected MAPE ~18.889%, got {mape}"

    # Test epsilon zero-division guard
    y_zero = np.array([0.0, 10.0])
    y_p_zero = np.array([1.0, 12.0])
    mape_zero = compute_mape(y_zero, y_p_zero, eps=1e-4)
    assert np.isfinite(mape_zero), "compute_mape must not return NaN or Inf when y_true contains zero"


def test_mase_scaling_invariance():
    """
    [15 Points] Verifies MASE formulation, benchmark hurdle equivalence, and scale invariance.
    """
    assert callable(compute_mase), "compute_mase missing"

    y_train = np.array([10.0, 12.0, 14.0, 16.0])  # diffs are all 2.0 -> denom = 2.0
    y_test = np.array([18.0, 20.0])
    # Naive forecast from y_train[-1] is 16.0: errors are [18-16=2.0, 20-16=4.0] -> test MAE = 3.0
    y_pred_naive = np.array([16.0, 16.0])
    mase_naive = compute_mase(y_train, y_test, y_pred_naive, m=1)
    assert np.isclose(mase_naive, 3.0 / 2.0), f"Expected MASE 1.5, got {mase_naive}"

    # Benchmark hurdle test: if test errors equal in-sample one-step Naive variation, MASE == 1.0
    y_pred_hurdle = np.array([16.0, 18.0])  # errors are [2.0, 2.0] -> test MAE = 2.0 -> MASE = 1.0
    assert np.isclose(compute_mase(y_train, y_test, y_pred_hurdle, m=1), 1.0), "MASE must equal 1.0 when test MAE equals in-sample Naive MAE"

    # Scale invariance test: multiply series by 1,000,000; MASE must be identical
    mase_scaled = compute_mase(y_train * 1e6, y_test * 1e6, y_pred_naive * 1e6, m=1)
    assert np.isclose(mase_naive, mase_scaled), "MASE must be strictly scale-invariant"


def test_distributional_scoring():
    """
    [15 Points] Verifies Pinball Loss asymmetric slopes and Winkler Interval scoring.
    """
    assert callable(compute_pinball_loss), "compute_pinball_loss missing"
    assert callable(compute_winkler_score), "compute_winkler_score missing"

    # 1. Pinball loss symmetry at alpha=0.50 (must equal 0.5 * MAE)
    y_true = np.array([10.0, 20.0])
    q_pred = np.array([12.0, 15.0])
    p_loss_half = compute_pinball_loss(y_true, q_pred, alpha=0.5)
    mae = compute_mae(y_true, q_pred)
    assert np.isclose(p_loss_half, 0.5 * mae), "Pinball loss at alpha=0.5 must equal 0.5 * MAE"

    # 2. Pinball loss asymmetry: alpha = 0.90 penalizes under-forecast (y > q) 9x more than over-forecast
    y_under = np.array([20.0])
    q_under = np.array([10.0])  # u = +10 -> loss = 0.90 * 10 = 9.0
    y_over = np.array([10.0])
    q_over = np.array([20.0])   # u = -10 -> loss = 0.10 * 10 = 1.0
    assert np.isclose(compute_pinball_loss(y_under, q_under, alpha=0.9), 9.0)
    assert np.isclose(compute_pinball_loss(y_over, q_over, alpha=0.9), 1.0)

    # 3. Winkler Score: within bounds [L, U], score is width U - L
    y_inside = np.array([15.0])
    score_inside = compute_winkler_score(y_inside, lower=10.0, upper=20.0, alpha=0.05)
    assert np.isclose(score_inside, 10.0), f"Expected Winkler width 10.0, got {score_inside}"

    # Breach lower bound by 2.0 units: penalty is (2 / 0.05) * 2 = 80 -> total = 10 + 80 = 90
    y_breach = np.array([8.0])
    score_breach = compute_winkler_score(y_breach, lower=10.0, upper=20.0, alpha=0.05)
    assert np.isclose(score_breach, 90.0), f"Expected Winkler breach score 90.0, got {score_breach}"


def test_walk_forward_split_engine():
    """
    [15 Points] Verifies expanding and rolling walk-forward split generation.
    """
    assert callable(generate_walk_forward_splits), "generate_walk_forward_splits missing"

    n_samples = 20
    min_train = 10
    horizon = 3
    step = 2

    splits_exp = generate_walk_forward_splits(n_samples, min_train, horizon, step, mode="expanding")
    assert len(splits_exp) >= 3, "Must generate at least 3 splits"

    for i, (train_idx, test_idx) in enumerate(splits_exp):
        assert len(test_idx) == horizon, f"Test horizon must be {horizon}"
        assert len(train_idx) == min_train + i * step, "Expanding train size must grow with step"
        assert train_idx[-1] < test_idx[0], "Temporal boundary violation: train must strictly precede test"
        assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and test must be disjoint"

    # Rolling window mode
    splits_rol = generate_walk_forward_splits(n_samples, min_train, horizon, step, mode="rolling")
    for train_idx, test_idx in splits_rol:
        assert len(train_idx) == min_train, "Rolling train size must remain fixed at min_train"


def test_kfold_leakage_proof():
    """
    [15 Points] Proves that random K-fold produces fraudulent R^2 on persistent AR(1) data.
    """
    assert callable(demonstrate_kfold_leakage), "demonstrate_kfold_leakage missing"

    np.random.seed(42)
    T = 120
    eps = np.random.normal(0, 1.0, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.95 * y[t - 1] + eps[t]

    res = demonstrate_kfold_leakage(pd.Series(y), n_splits=5)
    assert "kfold_r2" in res and "walk_forward_r2" in res and "leakage_gap" in res
    assert res["kfold_r2"] > 0.80, f"K-fold R2 should be fraudulently high (>0.80), got {res['kfold_r2']}"
    assert res["walk_forward_r2"] < 0.45, f"Walk-forward R2 should reflect true difficulty (<0.45), got {res['walk_forward_r2']}"
    assert res["leakage_gap"] > 0.40, f"Leakage gap must be substantial (>0.40), got {res['leakage_gap']}"


def test_diebold_mariano_hac_hln():
    """
    [15 Points] Verifies Diebold-Mariano test under H0, Newey-West HAC variance, and HLN correction.
    """
    assert callable(diebold_mariano_test), "diebold_mariano_test missing"

    np.random.seed(123)
    P = 60
    y_true = np.random.normal(100, 10, P)

    # 1. Null hypothesis: two identical models
    p_null1 = y_true + np.random.normal(0, 2, P)
    stat, pval, is_sig = diebold_mariano_test(y_true, p_null1, p_null1, h=1)
    assert np.isclose(stat, 0.0), "DM stat must be 0 for identical forecasts"
    assert np.isclose(pval, 1.0), "p-value must be 1.0 for identical forecasts"
    assert not is_sig, "Must not reject H0 for identical forecasts"

    # 2. Superior model: Model 1 has small error, Model 2 has large error
    p_good = y_true + np.random.normal(0, 0.5, P)
    p_bad = y_true + np.random.normal(0, 10.0, P)
    stat_sup, pval_sup, is_sig_sup = diebold_mariano_test(y_true, p_good, p_bad, h=1, small_sample=True)
    assert stat_sup < -2.0, "DM stat should be strongly negative (Model 1 superior)"
    assert pval_sup < 0.01, "p-value must reject equal accuracy"
    assert is_sig_sup, "Must reject H0 when one model is overwhelmingly superior"

    # 3. Test multi-step autocorrelation adjustment (h=4)
    stat_h4, pval_h4, _ = diebold_mariano_test(y_true, p_good, p_bad, h=4, small_sample=True)
    assert np.isfinite(stat_h4) and np.isfinite(pval_h4), "Multi-step DM test must compute finite statistics"
