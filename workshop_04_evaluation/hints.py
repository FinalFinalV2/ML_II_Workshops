#!/usr/bin/env python3
"""
hints.py — Step-by-Step Implementation Hints for Workshop 04
Course: Machine Learning II: Forecasting Economic Time Series
Topic: Evaluating Forecasts, Validation Schemes & Statistical Testing

Usage:
    Terminal:
        python hints.py                   # Interactive menu
        python hints.py --list            # List all tasks
        python hints.py --task <name>     # Show line-by-line hints for a specific task
        python hints.py --all             # Show all hints sequentially

    Jupyter Notebook:
        from hints import hint, list_hints
        list_hints()
        hint("diebold_mariano_test")
"""

import sys
import argparse
from typing import Dict, List, Optional

# ANSI Color codes for clean terminal presentation
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


HINTS: Dict[str, Dict[str, any]] = {
    "load_macro_series": {
        "title": "Milestone 1: Macroeconomic Data Ingestion (`load_macro_series`)",
        "signature": "def load_macro_series(filepath: str, date_col: str, val_col: str, freq: Optional[str] = None) -> pd.Series",
        "objective": "Ingests a raw macroeconomic CSV file, parses dates, removes duplicates, enforces a chronological DatetimeIndex with explicit frequency, and returns a clean float Series.",
        "line_by_line": [
            ("Line 1", "Read the CSV file into a pandas DataFrame:", "df = pd.read_csv(filepath)"),
            ("Line 2", "Parse the date column into proper datetime timestamps:", "df[date_col] = pd.to_datetime(df[date_col])"),
            ("Line 3", "Sort chronologically and eliminate duplicate timestamps:", "df = df.sort_values(date_col).drop_duplicates(subset=[date_col])"),
            ("Line 4", "Set the date column as the DataFrame index:", "df = df.set_index(date_col)"),
            ("Line 5", "Extract the target series as a float array:", "series = df[val_col].astype(float)"),
            ("Line 6", "Enforce the target calendar frequency (e.g., 'MS', 'QS', 'D') if provided:", "if freq is not None:\n    series = series.asfreq(freq)"),
            ("Line 7", "Assign series name and return:", "series.name = val_col\nreturn series"),
        ],
        "gotchas": [
            "Always sort chronologically before dropping duplicates or applying asfreq().",
            "asfreq() may introduce NaNs if periods are missing in the calendar; do not naively dropna() without investigating.",
            "Ensure the return object is a 1D pd.Series, not a 1-column DataFrame.",
        ],
        "verification": "Check: series.index is pd.DatetimeIndex, series.dtype is float, and series.index.is_monotonic_increasing is True."
    },

    "compute_mae": {
        "title": "Milestone 2a: Mean Absolute Error (`compute_mae`)",
        "signature": "def compute_mae(y_true, y_pred) -> float",
        "objective": "Computes Mean Absolute Error (MAE) under linear L1 loss: MAE = (1/N) * sum(|y_t - y_hat_t|).",
        "line_by_line": [
            ("Line 1", "Cast actual values to a 1D float numpy array:", "y_t = np.asarray(y_true, dtype=float)"),
            ("Line 2", "Cast predicted values to a 1D float numpy array:", "y_p = np.asarray(y_pred, dtype=float)"),
            ("Line 3", "Compute absolute differences and take the arithmetic mean:", "return float(np.mean(np.abs(y_t - y_p)))"),
        ],
        "gotchas": [
            "MAE assigns equal linear weight to all errors (a 10-unit miss costs exactly 10x a 1-unit miss).",
            "Bassett & Koenker theorem: The theoretical minimizer of MAE is the conditional median, not the conditional mean!",
            "Do not confuse with RMSE; MAE does not square errors.",
        ],
        "verification": "compute_mae([10, 20, 30], [12, 18, 33]) yields float((2 + 2 + 3) / 3) = 2.3333333333333335."
    },

    "compute_rmse": {
        "title": "Milestone 2b: Root Mean Squared Error (`compute_rmse`)",
        "signature": "def compute_rmse(y_true, y_pred) -> float",
        "objective": "Computes Root Mean Squared Error (RMSE) under quadratic L2 loss: RMSE = sqrt((1/N) * sum((y_t - y_hat_t)^2)).",
        "line_by_line": [
            ("Line 1", "Cast actual values to a float array:", "y_t = np.asarray(y_true, dtype=float)"),
            ("Line 2", "Cast predictions to a float array:", "y_p = np.asarray(y_pred, dtype=float)"),
            ("Line 3", "Compute mean squared error and take the square root:", "return float(np.sqrt(np.mean((y_t - y_p) ** 2)))"),
        ],
        "gotchas": [
            "RMSE heavily penalizes large outlier errors because errors are squared before averaging.",
            "The theoretical optimal point forecast that minimizes expected RMSE is the conditional mean.",
            "RMSE is always >= MAE. The ratio RMSE / MAE increases as the error distribution develops fatter tails.",
        ],
        "verification": "compute_rmse([10, 20, 30], [10, 20, 40]) yields sqrt(100 / 3) = 5.773502691896258."
    },

    "compute_mape": {
        "title": "Milestone 2c: Mean Absolute Percentage Error (`compute_mape`)",
        "signature": "def compute_mape(y_true, y_pred, eps: float = 1e-5) -> float",
        "objective": "Computes Mean Absolute Percentage Error (MAPE) with an epsilon safeguard against division by zero.",
        "line_by_line": [
            ("Line 1", "Cast actual values to a float array:", "y_t = np.asarray(y_true, dtype=float)"),
            ("Line 2", "Cast predictions to a float array:", "y_p = np.asarray(y_pred, dtype=float)"),
            ("Line 3", "Safeguard denominator: replace values where |y_t| < eps with eps:", "denom = np.where(np.abs(y_t) < eps, eps, np.abs(y_t))"),
            ("Line 4", "Calculate absolute percentage errors for each observation:", "pct_errors = np.abs((y_t - y_p) / denom) * 100.0"),
            ("Line 5", "Return the mean percentage error across all observations:", "return float(np.mean(pct_errors))"),
        ],
        "gotchas": [
            "Asymmetric penalty: An over-prediction of 100 on actual 10 gives 900% error, but an under-prediction of 10 on actual 100 gives only 10% error.",
            "Never use MAPE on series with negative or zero values (e.g., inflation rates, temperatures, net profits).",
            "Without the epsilon check, any actual value near 0 causes MAPE to explode to infinity.",
        ],
        "verification": "compute_mape([100, 200], [110, 190]) yields (10% + 5%) / 2 = 7.5%."
    },

    "compute_mase": {
        "title": "Milestone 3: Mean Absolute Scaled Error (`compute_mase`)",
        "signature": "def compute_mase(y_train, y_test, y_pred, m: int = 1) -> float",
        "objective": "Computes MASE (Hyndman & Koehler, 2006). Evaluates out-of-sample MAE relative to the in-sample mean absolute error of a 1-step benchmark (Naive for m=1, Seasonal Naive for m>1).",
        "line_by_line": [
            ("Line 1", "Cast training, test, and prediction inputs to float arrays:", "y_tr = np.asarray(y_train, dtype=float)\ny_te = np.asarray(y_test, dtype=float)\ny_p = np.asarray(y_pred, dtype=float)"),
            ("Line 2", "Compute benchmark in-sample denominator for non-seasonal (m=1):", "if m == 1:\n    denom = np.mean(np.abs(np.diff(y_tr)))"),
            ("Line 3", "Compute benchmark in-sample denominator for seasonal (m>1):", "else:\n    if len(y_tr) <= m:\n        raise ValueError(f'Training series length ({len(y_tr)}) must exceed seasonal period m ({m})')\n    denom = np.mean(np.abs(y_tr[m:] - y_tr[:-m]))"),
            ("Line 4", "Guard against zero benchmark variation (constant historical series):", "if denom == 0.0 or np.isnan(denom):\n    raise ValueError('In-sample benchmark variation denominator is zero (constant series).')"),
            ("Line 5", "Compute test set Mean Absolute Error:", "mae_test = np.mean(np.abs(y_te - y_p))"),
            ("Line 6", "Scale test MAE by in-sample benchmark denominator and return:", "return float(mae_test / denom)"),
        ],
        "gotchas": [
            "CRITICAL: The scale denominator MUST be computed strictly on y_train! Never compute it on y_test.",
            "MASE < 1.0 means your model outperforms the in-sample naive persistence benchmark.",
            "MASE > 1.0 means your model is worse than simply guessing the previous observation.",
            "Scale-free: MASE can be compared across completely different time series (e.g. GDP in billions vs. interest rates in %).",
        ],
        "verification": "If y_train=[10, 12, 14, 16], denom=2.0. If test actuals=[18, 20] and preds=[19, 21], mae_test=1.0 -> MASE = 0.50."
    },

    "compute_pinball_loss": {
        "title": "Milestone 4a: Quantile / Pinball Loss (`compute_pinball_loss`)",
        "signature": "def compute_pinball_loss(y_true, q_pred, alpha: float) -> float",
        "objective": "Computes asymmetric quantile loss for target quantile alpha in (0, 1). Penalizes under-prediction by slope alpha and over-prediction by slope (1 - alpha).",
        "line_by_line": [
            ("Line 1", "Validate that alpha is strictly within the open interval (0, 1):", "if not (0.0 < alpha < 1.0):\n    raise ValueError('alpha must be strictly within (0, 1)')"),
            ("Line 2", "Cast inputs to float numpy arrays:", "y_t = np.asarray(y_true, dtype=float)\nq_p = np.asarray(q_pred, dtype=float)"),
            ("Line 3", "Calculate the signed forecast error (actual - predicted_quantile):", "u = y_t - q_p"),
            ("Line 4", "Apply asymmetric tilted loss function: max(alpha * u, (alpha - 1) * u):", "loss = np.maximum(alpha * u, (alpha - 1.0) * u)"),
            ("Line 5", "Average across all observations and return scalar float:", "return float(np.mean(loss))"),
        ],
        "gotchas": [
            "Pinball loss is strictly proper: minimizing expected loss recovers the true conditional alpha-quantile.",
            "When alpha = 0.5, pinball loss is exactly 0.5 * MAE (median regression).",
            "At alpha = 0.90, under-prediction (actual > forecast) has slope 0.90, whereas over-prediction has slope 0.10 (a 9:1 penalty ratio).",
        ],
        "verification": "For alpha=0.9: y=20, q=10 -> u=+10 -> loss = 0.9*10 = 9.0. If y=20, q=30 -> u=-10 -> loss = (0.9-1.0)*(-10) = 1.0."
    },

    "compute_winkler_score": {
        "title": "Milestone 4b: Winkler Interval Score (`compute_winkler_score`)",
        "signature": "def compute_winkler_score(y_true, lower, upper, alpha: float = 0.05) -> float",
        "objective": "Evaluates central (1 - alpha) prediction intervals [lower, upper]. Balances interval width (sharpness) against miss penalties (calibration).",
        "line_by_line": [
            ("Line 1", "Validate nominal miss rate alpha in (0, 1):", "if not (0.0 < alpha < 1.0):\n    raise ValueError('alpha must be strictly within (0, 1)')"),
            ("Line 2", "Cast inputs to float arrays:", "y_t = np.asarray(y_true, dtype=float)\nl = np.asarray(lower, dtype=float)\nu = np.asarray(upper, dtype=float)"),
            ("Line 3", "Compute base sharpness cost (interval width):", "width = u - l"),
            ("Line 4", "Compute lower breach penalty when y < lower:", "lower_pen = np.where(y_t < l, (2.0 / alpha) * (l - y_t), 0.0)"),
            ("Line 5", "Compute upper breach penalty when y > upper:", "upper_pen = np.where(y_t > u, (2.0 / alpha) * (y_t - u), 0.0)"),
            ("Line 6", "Sum base width and penalties:", "score = width + lower_pen + upper_pen"),
            ("Line 7", "Average across all observations and return:", "return float(np.mean(score))"),
        ],
        "gotchas": [
            "If actual is inside [lower, upper], penalty is 0, so the score is simply the interval width (narrower is better).",
            "For a 95% interval (alpha = 0.05), the miss penalty slope is 2.0 / 0.05 = 40x the distance of the miss!",
            "Winkler score is strictly proper: minimizing expected score produces calibrated central prediction intervals.",
        ],
        "verification": "For alpha=0.10 (90% interval), [10, 20]: hit at y=15 -> score=10.0. Miss at y=25 -> penalty=(2/0.1)*5=100 -> score=110.0."
    },

    "generate_walk_forward_splits": {
        "title": "Milestone 5: Walk-Forward Cross-Validation Engine (`generate_walk_forward_splits`)",
        "signature": "def generate_walk_forward_splits(n_samples: int, min_train: int, forecast_horizon: int, step: int = 1, mode: str = 'expanding') -> List[Tuple[np.ndarray, np.ndarray]]",
        "objective": "Generates chronological (train_idx, test_idx) tuples for out-of-sample backtesting without lookahead leakage. Supports 'expanding' and 'rolling' modes.",
        "line_by_line": [
            ("Line 1", "Validate that sample size accommodates at least one full train+test window:", "if min_train + forecast_horizon > n_samples:\n    raise ValueError('min_train + forecast_horizon exceeds total sample size')"),
            ("Line 2", "Initialize empty split list and set origin at min_train:", "splits = []\norigin = min_train"),
            ("Line 3", "Loop while the test window does not exceed n_samples:", "while origin + forecast_horizon <= n_samples:"),
            ("Line 4", "Build train indices according to selected windowing mode:", "    if mode == 'expanding':\n        train_idx = np.arange(0, origin)\n    elif mode == 'rolling':\n        train_idx = np.arange(origin - min_train, origin)\n    else:\n        raise ValueError(f'Unknown mode: {mode}. Must be \"expanding\" or \"rolling\"')"),
            ("Line 5", "Construct test index slice strictly from origin to origin + forecast_horizon:", "    test_idx = np.arange(origin, origin + forecast_horizon)"),
            ("Line 6", "Append tuple to splits list and advance origin by step:", "    splits.append((train_idx, test_idx))\n    origin += step"),
            ("Line 7", "Return complete list of splits:", "return splits"),
        ],
        "gotchas": [
            "Expanding mode: train window grows (0 to origin). Rolling mode: train window slides (origin - min_train to origin).",
            "train_idx max is origin - 1; test_idx min is origin. They must NEVER overlap (zero leakage).",
            "The number of splits is floor((n_samples - min_train - forecast_horizon) / step) + 1.",
        ],
        "verification": "generate_walk_forward_splits(100, 50, 10, step=10) produces exactly 5 splits, each with train length >= 50 and test length 10."
    },

    "demonstrate_kfold_leakage": {
        "title": "Milestone 6: Proving the Random K-Fold Leakage Crime (`demonstrate_kfold_leakage`)",
        "signature": "def demonstrate_kfold_leakage(series: pd.Series, n_splits: int = 5, horizon: int = 12) -> Dict[str, float]",
        "objective": "Demonstrates why random K-Fold cross-validation produces fraudulent, over-optimistic R^2 metrics on autoregressive series compared to honest walk-forward evaluation.",
        "line_by_line": [
            ("Line 1", "Construct 1-step lag regression dataset (X_t = y_{t-1}, Y_t = y_t):", "y = np.asarray(series, dtype=float)\nX_lag = y[:-1].reshape(-1, 1)\ny_target = y[1:]\nN = len(y_target)"),
            ("Line 2", "Execute Shuffled K-Fold CV (the temporal crime):", "kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\nkf_preds = np.zeros(N)\nfor trn, tst in kf.split(X_lag):\n    reg = LinearRegression().fit(X_lag[trn], y_target[trn])\n    kf_preds[tst] = reg.predict(X_lag[tst])\nss_tot_kf = np.sum((y_target - np.mean(y_target)) ** 2)\nkf_r2 = float(1.0 - np.sum((y_target - kf_preds) ** 2) / ss_tot_kf)"),
            ("Line 3", "Execute Chronological Multi-Step Walk-Forward (honest backtest):", "tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)\nwf_preds, wf_actuals = [], []\nfor trn, tst in tscv.split(y):\n    reg = LinearRegression().fit(y[trn[:-1]].reshape(-1, 1), y[trn[1:]])\n    curr = y[trn[-1]]\n    for _ in range(len(tst)):\n        curr = float(reg.predict([[curr]])[0])\n        wf_preds.append(curr)\n    wf_actuals.extend(y[tst])"),
            ("Line 4", "Compute authentic Walk-Forward R2:", "wf_actuals = np.array(wf_actuals)\nwf_preds = np.array(wf_preds)\nss_tot_wf = np.sum((wf_actuals - np.mean(wf_actuals)) ** 2)\nwf_r2 = float(1.0 - np.sum((wf_actuals - wf_preds) ** 2) / ss_tot_wf)"),
            ("Line 5", "Package and return metrics dictionary:", "return {\n    'kfold_r2': kf_r2,\n    'walk_forward_r2': wf_r2,\n    'leakage_gap': float(kf_r2 - wf_r2)\n}"),
        ],
        "gotchas": [
            "In shuffled K-Fold, future points condition past points (predicting t=49 using t=48 when t=50 was in training).",
            "Because persistent macroeconomic series have high autocorrelation, shuffled K-Fold reports R^2 > 0.90.",
            "In reality, honest multi-step recursive walk-forward yields an R^2 near 0.10 to 0.30.",
            "The difference ('leakage_gap') represents purely fictitious performance.",
        ],
        "verification": "On persistent series like CPI, kf_r2 is typically > 0.90, wf_r2 is < 0.30, and leakage_gap > 0.60."
    },

    "evaluate_horizon_degradation": {
        "title": "Milestone 7: Horizon Degradation Profiles (`evaluate_horizon_degradation`)",
        "signature": "def evaluate_horizon_degradation(y: pd.Series, model_fn: Callable, min_train: int, max_h: int) -> pd.DataFrame",
        "objective": "Evaluates how forecast error compounds as lead time h increases from 1 to max_h across all rolling origins.",
        "line_by_line": [
            ("Line 1", "Cast series to float array and initialize error containers by horizon:", "y_arr = np.asarray(y, dtype=float)\nn = len(y_arr)\nerrors_by_h = {h: [] for h in range(1, max_h + 1)}"),
            ("Line 2", "Roll origin through time from min_train until origin + max_h exceeds n:", "origin = min_train\nwhile origin + max_h <= n:\n    train = y_arr[:origin]\n    actuals = y_arr[origin:origin + max_h]\n    preds = model_fn(train, steps=max_h)\n    for h in range(1, max_h + 1):\n        errors_by_h[h].append(actuals[h - 1] - preds[h - 1])\n    origin += 1"),
            ("Line 3", "Aggregate MAE and RMSE across all origins for each horizon h:", "records = []\nfor h in range(1, max_h + 1):\n    errs = np.array(errors_by_h[h])\n    mae = float(np.mean(np.abs(errs)))\n    rmse = float(np.sqrt(np.mean(errs ** 2)))\n    records.append({'horizon': h, 'mae': mae, 'rmse': rmse})"),
            ("Line 4", "Return indexed DataFrame:", "return pd.DataFrame(records).set_index('horizon')"),
        ],
        "gotchas": [
            "Errors should monotonically increase with horizon h: predicting 12 months ahead is harder than 1 month ahead.",
            "model_fn must accept (train, steps) and return a 1D array of forecasts of length steps.",
            "Remember that h is 1-indexed, but arrays are 0-indexed (actuals[h - 1] - preds[h - 1]).",
        ],
        "verification": "The resulting DataFrame has index 1..max_h and columns ['mae', 'rmse']. Verify df.loc[max_h, 'rmse'] >= df.loc[1, 'rmse']."
    },

    "diebold_mariano_test": {
        "title": "Milestone 8: Diebold-Mariano Hypothesis Test (`diebold_mariano_test`)",
        "signature": "def diebold_mariano_test(y_true, y_pred1, y_pred2, h: int = 1, loss: str = 'squared', small_sample: bool = True) -> Tuple[float, float, bool]",
        "objective": "Statistically tests whether the accuracy difference between two forecasts is significant under serially correlated errors. Uses Newey-West HAC variance with Bartlett kernel and Harvey-Leybourne-Newbold (HLN 1997) finite-sample correction.",
        "line_by_line": [
            ("Line 1", "Cast inputs to float arrays and compute forecast error series:", "y_t = np.asarray(y_true, dtype=float)\np1 = np.asarray(y_pred1, dtype=float)\np2 = np.asarray(y_pred2, dtype=float)\ne1 = y_t - p1\ne2 = y_t - p2"),
            ("Line 2", "Compute loss differential series d_t under chosen loss function:", "if loss == 'squared':\n    d = e1 ** 2 - e2 ** 2\nelif loss == 'absolute':\n    d = np.abs(e1) - np.abs(e2)\nelse:\n    raise ValueError(f'Unknown loss: {loss}. Choose \"squared\" or \"absolute\"')"),
            ("Line 3", "Validate sample size P >= 3:", "P = len(d)\nif P < 3:\n    raise ValueError('Sample size P must be >= 3 for Diebold-Mariano test.')"),
            ("Line 4", "Compute sample mean differential and contemporaneous variance gamma_0:", "d_bar = np.mean(d)\ngamma_0 = np.var(d, ddof=0)"),
            ("Line 5", "Compute Newey-West HAC long-run variance with Bartlett triangular kernel up to lag h-1:", "long_run_var = gamma_0\nfor k in range(1, h):\n    if k < P:\n        cov_k = np.cov(d[k:], d[:-k], ddof=0)[0, 1]\n        weight = 1.0 - (k / h)\n        long_run_var += 2.0 * weight * cov_k"),
            ("Line 6", "Compute variance of d_bar with numerical stability protection:", "var_d_bar = long_run_var / P\nif var_d_bar <= 0:\n    var_d_bar = max(1e-12, gamma_0 / P)\ndm_stat = d_bar / np.sqrt(var_d_bar)"),
            ("Line 7", "Apply Harvey-Leybourne-Newbold (1997) finite-sample correction:", "if small_sample:\n    hln_factor = np.sqrt((P + 1 - 2 * h + (h * (h - 1)) / P) / P)\n    test_stat = float(dm_stat * hln_factor)\n    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(test_stat), df=P - 1)))\nelse:\n    test_stat = float(dm_stat)\n    p_value = float(2.0 * (1.0 - stats.norm.cdf(np.abs(test_stat))))"),
            ("Line 8", "Evaluate significance at 5% alpha and return tuple:", "is_significant = bool(p_value < 0.05)\nreturn test_stat, p_value, is_significant"),
        ],
        "gotchas": [
            "For h=1 (one-step forecasts), errors are serially uncorrelated; the HAC loop does not run, and variance is just gamma_0 / P.",
            "For h > 1, optimal h-step forecast errors follow an MA(h-1) process. Ignoring serial correlation inflates the test statistic and produces severe false positive rates!",
            "Sign interpretation: If test_stat < 0, Model 1 has lower loss (Model 1 wins). If test_stat > 0, Model 2 wins.",
            "Use Student-t distribution with df = P - 1 when small_sample=True, not N(0, 1).",
        ],
        "verification": "If y_pred1 == y_pred2, d_t = 0 everywhere, resulting in test_stat = 0.0, p_value = 1.0, is_significant = False."
    },

    "run_leakage_audit_suite": {
        "title": "Milestone 9: Temporal Leakage Audit Suite (`run_leakage_audit_suite`)",
        "signature": "def run_leakage_audit_suite(train_features: pd.DataFrame, test_features: pd.DataFrame, model: Any = None) -> Dict[str, bool]",
        "objective": "Production verification suite to confirm that training and test splits adhere strictly to temporal causality with zero data contamination.",
        "line_by_line": [
            ("Line 1", "Initialize audit dictionary:", "audit = {}"),
            ("Line 2", "Check 1: Strict chronological boundary (train max timestamp strictly < test min timestamp):", "if isinstance(train_features.index, pd.DatetimeIndex) and isinstance(test_features.index, pd.DatetimeIndex):\n    audit['strict_chronological_boundary'] = bool(train_features.index.max() < test_features.index.min())\nelse:\n    audit['strict_chronological_boundary'] = True"),
            ("Line 3", "Check 2: Disjoint partitions (no shared index keys or timestamps):", "audit['disjoint_partitions'] = bool(len(set(train_features.index).intersection(set(test_features.index))) == 0)"),
            ("Line 4", "Check 3: Feature column alignment (exact column names and order):", "audit['feature_columns_aligned'] = bool(list(train_features.columns) == list(test_features.columns))"),
            ("Line 5", "Return audit summary dictionary:", "return audit"),
        ],
        "gotchas": [
            "A single overlapping timestamp between train and test introduces future information leakage.",
            "Never fit scalers (e.g. StandardScaler) on the full dataset before splitting; fit only on train and transform test.",
            "All checks must evaluate to True before deploying any forecasting model.",
        ],
        "verification": "When passing split DataFrame partitions, all 3 dictionary values must be True."
    }
}


def print_banner():
    print(f"\n{BLUE}{'═'*78}{RESET}")
    print(f"{BOLD}{CYAN}  Workshop 04: Evaluating Forecasts & Walk-Forward Validation  {RESET}")
    print(f"{DIM}  Pure Python Implementation Hints & Step-by-Step Blueprints  {RESET}")
    print(f"{BLUE}{'═'*78}{RESET}\n")


def print_task_hint(task_name: str):
    if task_name not in HINTS:
        print(f"{YELLOW}Task '{task_name}' not found. Available tasks:{RESET}")
        for k in HINTS:
            print(f"  - {k}")
        return

    h = HINTS[task_name]
    print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────{RESET}")
    print(f"{BOLD}{GREEN}{h['title']}{RESET}")
    print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────{RESET}")
    print(f"{BOLD}Signature:{RESET}\n  {YELLOW}{h['signature']}{RESET}\n")
    print(f"{BOLD}Objective:{RESET}\n  {h['objective']}\n")
    print(f"{BOLD}Step-by-Step Blueprint (Line-by-Line):{RESET}")

    for step_num, desc, code in h["line_by_line"]:
        print(f"  {CYAN}▸ {BOLD}{step_num}:{RESET} {desc}")
        for code_line in code.split("\n"):
            print(f"    {BLUE}>>> {code_line}{RESET}")
    print()

    print(f"{BOLD}Common Gotchas & Pitfalls:{RESET}")
    for g in h["gotchas"]:
        print(f"  {MAGENTA}⚠️  {g}{RESET}")
    print()

    print(f"{BOLD}Self-Check:{RESET} {h['verification']}")
    print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────{RESET}\n")


def list_hints():
    """Prints list of all available tasks in Workshop 04."""
    print_banner()
    print(f"{BOLD}Available Workshop 04 Tasks:{RESET}")
    for i, (k, v) in enumerate(HINTS.items(), 1):
        print(f"  {CYAN}{i:2d}.{RESET} {BOLD}{k:<30}{RESET} {DIM}— {v['title'].split(':')[1].strip()}{RESET}")
    print(f"\nRun {GREEN}python hints.py --task <task_name>{RESET} or in Jupyter: {GREEN}hint('<task_name>'){RESET}\n")


def hint(task_name: str):
    """Jupyter-friendly function to inspect a specific task hint."""
    print_task_hint(task_name)


def show_all_hints():
    """Prints all hints sequentially."""
    print_banner()
    for task_name in HINTS:
        print_task_hint(task_name)


def main():
    parser = argparse.ArgumentParser(description="Workshop 04 Line-by-Line Hints")
    parser.add_argument("--task", type=str, help="Specific task name (e.g. diebold_mariano_test, compute_mase)")
    parser.add_argument("--list", action="store_true", help="List all available tasks")
    parser.add_argument("--all", action="store_true", help="Print all hints sequentially")
    args = parser.parse_args()

    if args.list:
        list_hints()
    elif args.all:
        show_all_hints()
    elif args.task:
        print_banner()
        print_task_hint(args.task)
    else:
        # Default interactive menu
        list_hints()
        try:
            choice = input(f"{BOLD}Enter task name or number to view (or press Enter to exit): {RESET}").strip()
            if not choice:
                return
            if choice.isdigit():
                idx = int(choice) - 1
                keys = list(HINTS.keys())
                if 0 <= idx < len(keys):
                    print_task_hint(keys[idx])
                else:
                    print(f"{YELLOW}Invalid number.{RESET}")
            else:
                print_task_hint(choice)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")


if __name__ == "__main__":
    main()
