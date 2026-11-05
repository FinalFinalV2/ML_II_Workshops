#!/usr/bin/env python3
"""
hints.py — Step-by-Step Implementation Hints for Workshop 03
Course: Machine Learning II: Forecasting Economic Time Series
Topic: The Forecaster's Toolbox — Benchmarks, Residual Diagnostics & Prediction Intervals

Usage:
    Terminal:
        python hints.py                   # Interactive menu
        python hints.py --list            # List all tasks
        python hints.py --task <name>     # Show line-by-line hints for a specific task
        python hints.py --all             # Show all hints sequentially

    Jupyter Notebook:
        from hints import hint, list_hints
        list_hints()
        hint("forecast_snaive")
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


HINTS: Dict[str, Dict[str, str]] = {
    "load_and_index_series": {
        "title": "Task 1: Load and Index Economic Time Series (`load_and_index_series`)",
        "signature": "def load_and_index_series(filepath: str, date_col: str, val_col: str, freq: Optional[str] = None) -> pd.Series",
        "objective": "Ingests a raw CSV file, enforces an immutable DatetimeIndex, sets calendar frequency, and returns a clean float Series.",
        "line_by_line": [
            ("Line 1", "Load the raw CSV into a DataFrame:", "df = pd.read_csv(filepath)"),
            ("Line 2", "Parse the date column into datetime objects:", "df[date_col] = pd.to_datetime(df[date_col])"),
            ("Line 3", "Set the date column as the index and sort chronologically:", "df = df.set_index(date_col).sort_index()"),
            ("Line 4", "Extract the target column as float:", "series = df[val_col].astype(float)"),
            ("Line 5", "Enforce explicit calendar frequency:", "if freq is not None:\n    series = series.asfreq(freq)\nelif pd.infer_freq(series.index) is not None:\n    series = series.asfreq(pd.infer_freq(series.index))"),
            ("Line 6", "Set the series name and return:", "series.name = val_col\nreturn series"),
        ],
        "gotchas": [
            "Always sort the index chronologically before setting frequency.",
            "Do not drop NaNs created by asfreq() if the series has legitimate calendar gaps; let downstream metrics handle them or document imputation.",
            "Ensure the return object is a pd.Series, not a 1-column pd.DataFrame.",
        ],
        "verification": "Check: series.index is pd.DatetimeIndex and series.index.freq is not None."
    },

    "_generate_future_index": {
        "title": "Task 2: Future DatetimeIndex Generator (`_generate_future_index`)",
        "signature": "def _generate_future_index(y_train: pd.Series, h: int) -> pd.DatetimeIndex",
        "objective": "Creates an out-of-sample DatetimeIndex of length h starting strictly at T+1 (the period immediately following history).",
        "line_by_line": [
            ("Line 1", "Extract or infer calendar frequency from training index:", "freq = y_train.index.freq\nif freq is None:\n    inferred = pd.infer_freq(y_train.index)\n    freq = inferred if inferred is not None else 'D'"),
            ("Line 2", "Locate the final historical timestamp:", "last_date = y_train.index[-1]"),
            ("Line 3", "Generate h+1 dates starting from last_date:", "full_range = pd.date_range(start=last_date, periods=h + 1, freq=freq)"),
            ("Line 4", "Drop the first date (last_date itself) to leave periods T+1 through T+h:", "return full_range[1:]"),
        ],
        "gotchas": [
            "If start=last_date, period 0 is last_date (historical). You must slice [1:] to avoid lookahead/overlap with the training sample.",
            "Must return an exact length of h.",
        ],
        "verification": "assert len(_generate_future_index(y, 12)) == 12 and _generate_future_index(y, 12)[0] > y.index[-1]"
    },

    "forecast_mean": {
        "title": "Task 3: Mean Forecast Benchmark (`forecast_mean`)",
        "signature": "def forecast_mean(y_train: pd.Series, h: int) -> pd.Series",
        "objective": "Projects the historical sample mean horizontally across all future horizons: y_hat_{T+h|T} = (1/T) * sum(y_t).",
        "line_by_line": [
            ("Line 1", "Validate forecast horizon h is a positive integer:", "if h <= 0:\n    raise ValueError('Forecast horizon h must be a positive integer')"),
            ("Line 2", "Compute the scalar sample mean of the training data:", "mean_val = float(y_train.mean())"),
            ("Line 3", "Generate the future DatetimeIndex using our helper:", "idx = _generate_future_index(y_train, h)"),
            ("Line 4", "Construct and return the forecasted Series:", "return pd.Series(mean_val, index=idx, name=y_train.name)"),
        ],
        "gotchas": [
            "The forecast is flat (constant) across all h horizons.",
            "The index must strictly match the future DatetimeIndex from _generate_future_index.",
        ],
        "verification": "Mean forecast on [10, 20, 30] for h=3 yields [20.0, 20.0, 20.0]."
    },

    "forecast_naive": {
        "title": "Task 4: Naive (Persistence) Benchmark (`forecast_naive`)",
        "signature": "def forecast_naive(y_train: pd.Series, h: int) -> pd.Series",
        "objective": "Sets all future forecasts to the final observed historical value: y_hat_{T+h|T} = y_T. Optimal for a pure Random Walk.",
        "line_by_line": [
            ("Line 1", "Validate horizon h > 0:", "if h <= 0:\n    raise ValueError('h must be a positive integer')"),
            ("Line 2", "Extract the very last historical value using positional indexing:", "naive_val = float(y_train.iloc[-1])"),
            ("Line 3", "Generate future DatetimeIndex of length h:", "idx = _generate_future_index(y_train, h)"),
            ("Line 4", "Broadcast the naive value across the future index:", "return pd.Series(naive_val, index=idx, name=y_train.name)"),
        ],
        "gotchas": [
            "Use .iloc[-1] to ensure you retrieve the chronologically latest value, not label-based index .loc[-1].",
            "This is the yardstick baseline: any machine learning model with MASE > 1.0 is worse than this 1-line forecast.",
        ],
        "verification": "Naive forecast on [10, 20, 42] for h=4 yields [42.0, 42.0, 42.0, 42.0]."
    },

    "forecast_snaive": {
        "title": "Task 5: Seasonal Naive Benchmark (`forecast_snaive`)",
        "signature": "def forecast_snaive(y_train: pd.Series, h: int, m: int) -> pd.Series",
        "objective": "Projects the last observed value from the identical season of the previous cycle: y_hat_{T+h|T} = y_{T+h - m*(k+1)}.",
        "line_by_line": [
            ("Line 1", "Validate inputs: h > 0 and 1 <= m <= len(y_train):", "if h <= 0 or m <= 0 or m > len(y_train):\n    raise ValueError('Invalid h or seasonal period m')"),
            ("Line 2", "Extract the final m observations representing the most recent full seasonal cycle:", "last_season = y_train.iloc[-m:].values"),
            ("Line 3", "Use modulo indexing to tile the seasonal pattern across all h future steps:", "fc_vals = [last_season[(i - 1) % m] for i in range(1, h + 1)]"),
            ("Line 4", "Generate future index of length h:", "idx = _generate_future_index(y_train, h)"),
            ("Line 5", "Construct and return the Series:", "return pd.Series(fc_vals, index=idx, name=y_train.name)"),
        ],
        "gotchas": [
            "Notice the index offset: for step i=1, (1 - 1) % m = 0, which correctly selects last_season[0].",
            "If h > m, the seasonal cycle repeats cyclically (modulo arithmetic handles this automatically).",
        ],
        "verification": "If last 4 quarters are [Q1=100, Q2=150, Q3=200, Q4=250], h=6 forecasts [100, 150, 200, 250, 100, 150]."
    },

    "forecast_drift": {
        "title": "Task 6: Drift Benchmark (`forecast_drift`)",
        "signature": "def forecast_drift(y_train: pd.Series, h: int) -> pd.Series",
        "objective": "Extrapolates the historical linear trend joining the first (y_1) and last (y_T) observations: y_hat_{T+h|T} = y_T + h * (y_T - y_1) / (T - 1).",
        "line_by_line": [
            ("Line 1", "Validate h > 0 and sample size T >= 2:", "T = len(y_train)\nif h <= 0 or T < 2:\n    raise ValueError('h must be > 0 and T >= 2')"),
            ("Line 2", "Extract first and last historical points:", "y_1 = float(y_train.iloc[0])\ny_T = float(y_train.iloc[-1])"),
            ("Line 3", "Compute the average historical slope:", "slope = (y_T - y_1) / (T - 1)"),
            ("Line 4", "Create step vector [1, 2, ..., h]:", "steps = np.arange(1, h + 1)"),
            ("Line 5", "Compute forecast trajectory from y_T with compounding slope:", "fc_vals = y_T + steps * slope"),
            ("Line 6", "Return indexed Series:", "return pd.Series(fc_vals, index=_generate_future_index(y_train, h), name=y_train.name)"),
        ],
        "gotchas": [
            "Divide by (T - 1), NOT T, because there are T-1 transitions between T observations.",
            "If y_T == y_1, slope is 0.0 and drift gracefully collapses to naive forecast.",
        ],
        "verification": "On [10, 20, 30] (T=3, slope=(30-10)/2=10), h=2 yields [40.0, 50.0]."
    },

    "compute_benchmark_residuals": {
        "title": "Task 7: In-Sample Benchmark Residuals (`compute_benchmark_residuals`)",
        "signature": "def compute_benchmark_residuals(y_train: pd.Series, method: str, m: Optional[int] = None) -> pd.Series",
        "objective": "Computes one-step in-sample residuals e_t = y_t - y_hat_{t|t-1} across history without future leakage.",
        "line_by_line": [
            ("Line 1", "Compute fitted values based on method:", ""),
            ("      - for 'mean':", "fitted = y_train.expanding().mean().shift(1)  # average of y_1..y_{t-1}"),
            ("      - for 'naive':", "fitted = y_train.shift(1)  # yesterday's actual value"),
            ("      - for 'snaive':", "if m is None or m <= 0: raise ValueError('Seasonal period m required')\nfitted = y_train.shift(m)  # same season last year"),
            ("      - for 'drift':", "T = len(y_train)\nfitted = pd.Series(np.nan, index=y_train.index)\nfor t in range(1, T):\n    slope = (y_train.iloc[t-1] - y_train.iloc[0]) / (t - 1) if t > 1 else 0.0\n    fitted.iloc[t] = y_train.iloc[t-1] + slope"),
            ("Line 2", "Compute raw residuals:", "residuals = y_train - fitted"),
            ("Line 3", "Ensure Series name and index alignment:", "residuals.name = f'{method}_residuals'\nreturn residuals"),
        ],
        "gotchas": [
            "Always use .shift(1) or .shift(m). Never use contemporaneous y_t in fitted value y_hat_{t|t-1}.",
            "The first observation (or first m observations for snaive) will naturally be NaN. Do not drop them; retain index length.",
        ],
        "verification": "For naive on [10, 15, 12], fitted is [NaN, 10, 15] and residuals are [NaN, 5, -3]."
    },

    "compute_forecast_sd": {
        "title": "Task 8: Multi-Step Forecast Standard Deviation (`compute_forecast_sd`)",
        "signature": "def compute_forecast_sd(residuals: pd.Series, method: str, h: int, m: Optional[int] = None, T: Optional[int] = None) -> np.ndarray",
        "objective": "Implements the analytical multi-step standard error formulas from FPP3 Table 5.2.",
        "line_by_line": [
            ("Line 1", "Compute sample standard deviation of residuals (ddof=1):", "clean_res = residuals.dropna()\ns = float(clean_res.std(ddof=1))\nT = len(clean_res) if T is None else T\nsteps = np.arange(1, h + 1)"),
            ("Line 2", "Apply Table 5.2 formula based on method:", ""),
            ("      - for 'mean':", "sd = s * np.sqrt(1.0 + 1.0 / T) * np.ones(h)  # constant over h"),
            ("      - for 'naive':", "sd = s * np.sqrt(steps)  # expands with sqrt(h)"),
            ("      - for 'snaive':", "k = (steps - 1) // m\nsd = s * np.sqrt(k + 1.0)  # step function every m periods"),
            ("      - for 'drift':", "sd = s * np.sqrt(steps * (1.0 + steps / (T - 1.0)))  # compounds quadratically"),
            ("Line 3", "Return as numpy float array of length h:", "return np.asarray(sd, dtype=float)"),
        ],
        "gotchas": [
            "Use ddof=1 for sample standard deviation s.",
            "For snaive, integer division (steps - 1) // m creates the exact step-function uncertainty profile.",
        ],
        "verification": "For naive with s=2.0 and h=4: sd = [2*sqrt(1), 2*sqrt(2), 2*sqrt(3), 2*sqrt(4)] = [2.0, 2.828, 3.464, 4.0]."
    },

    "compute_prediction_intervals": {
        "title": "Task 9: Gaussian Prediction Intervals (`compute_prediction_intervals`)",
        "signature": "def compute_prediction_intervals(point_forecast: pd.Series, forecast_sd: np.ndarray, levels: List[int] = [80, 95]) -> pd.DataFrame",
        "objective": "Calculates symmetric Gaussian confidence intervals: [y_hat - z * sd, y_hat + z * sd].",
        "line_by_line": [
            ("Line 1", "Initialize results DataFrame with point forecast:", "df = pd.DataFrame({'point_forecast': point_forecast}, index=point_forecast.index)"),
            ("Line 2", "Iterate over requested coverage levels (e.g. 80, 95):", "for lvl in levels:\n    alpha = (100.0 - lvl) / 100.0\n    # Two-sided standard normal critical value z_{alpha/2}:\n    z = stats.norm.ppf(1.0 - alpha / 2.0)"),
            ("Line 3", "Compute lower and upper bounds:", "    margin = z * forecast_sd\n    df[f'lower_{lvl}'] = point_forecast - margin\n    df[f'upper_{lvl}'] = point_forecast + margin"),
            ("Line 4", "Return populated DataFrame:", "return df"),
        ],
        "gotchas": [
            "For 95% level, alpha = 0.05, and norm.ppf(0.975) ≈ 1.95996.",
            "For 80% level, alpha = 0.20, and norm.ppf(0.900) ≈ 1.28155.",
        ],
        "verification": "Check: df['lower_95'] < df['lower_80'] < df['point_forecast'] < df['upper_80'] < df['upper_95']."
    },

    "bootstrap_prediction_intervals": {
        "title": "Task 10: Bootstrapped Non-Gaussian Intervals (`bootstrap_prediction_intervals`)",
        "signature": "def bootstrap_prediction_intervals(y_train: pd.Series, method: str, h: int, m: Optional[int] = None, B: int = 1000, levels: List[int] = [80, 95], random_seed: int = 42) -> pd.DataFrame",
        "objective": "Simulates B future forecast paths by resampling historical residuals with replacement, requiring NO normality assumption.",
        "line_by_line": [
            ("Line 1", "Compute in-sample residuals and drop NaNs:", "res = compute_benchmark_residuals(y_train, method, m=m).dropna().values"),
            ("Line 2", "Seed random number generator for reproducibility:", "rng = np.random.default_rng(random_seed)"),
            ("Line 3", "Resample innovation matrix of shape (B, h) with replacement:", "sampled_errors = rng.choice(res, size=(B, h), replace=True)"),
            ("Line 4", "Simulate B future paths depending on benchmark dynamics:", ""),
            ("      - for 'naive':", "last_val = y_train.iloc[-1]\n# Cumulative sum of innovations simulates random walk paths:\npaths = last_val + np.cumsum(sampled_errors, axis=1)"),
            ("      - for 'drift':", "slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)\ndrift_line = y_train.iloc[-1] + np.arange(1, h + 1) * slope\npaths = drift_line + np.cumsum(sampled_errors, axis=1)"),
            ("Line 5", "Extract empirical percentiles for each coverage level:", "for lvl in levels:\n    alpha = (100.0 - lvl) / 100.0\n    df[f'lower_{lvl}'] = np.percentile(paths, (alpha / 2.0) * 100.0, axis=0)\n    df[f'upper_{lvl}'] = np.percentile(paths, (1.0 - alpha / 2.0) * 100.0, axis=0)"),
            ("Line 6", "Return DataFrame with point forecast and empirical bounds:", "return df"),
        ],
        "gotchas": [
            "Use np.cumsum along axis=1 because errors compound sequentially over lead time.",
            "Set explicit random_seed so test suites evaluate deterministically.",
        ],
        "verification": "Paths should be centered around the point forecast, with bounds that reflect empirical skewness."
    },

    "check_residuals": {
        "title": "Task 11: Automated Residual Diagnostic Battery (`check_residuals`)",
        "signature": "def check_residuals(residuals: pd.Series, lags: Optional[int] = None, seasonal_m: Optional[int] = None, K: int = 0) -> Dict[str, Any]",
        "objective": "Runs an automated 4-stage econometric forensic battery: Unbiasedness (t-test), Autocorrelation (Ljung-Box), Normality (Jarque-Bera), and Moments.",
        "line_by_line": [
            ("Line 1", "Clean residuals and get effective sample size:", "clean = residuals.dropna()\nT = len(clean)\nif T < 5: raise ValueError('Need at least 5 residuals')"),
            ("Line 2", "Test 1: Unbiasedness (H0: mean == 0):", "t_stat, p_unbiased = stats.ttest_1samp(clean, 0.0)\nis_unbiased = bool(p_unbiased > 0.05)"),
            ("Line 3", "Test 2: Ljung-Box Portmanteau Autocorrelation:", "if lags is None:\n    lags = min(2 * seasonal_m, T // 5) if seasonal_m else min(10, T // 5)\nlb_df = acorr_ljungbox(clean, lags=[lags], return_df=True)\nlb_stat = float(lb_df['lb_stat'].iloc[0])\n# Adjust degrees of freedom for fitted parameters K:\ndof = max(1, lags - K)\nlb_pvalue = float(stats.chi2.sf(lb_stat, dof))\nis_white_noise = bool(lb_pvalue > 0.05)"),
            ("Line 4", "Test 3: Skewness, Kurtosis & Jarque-Bera Normality:", "skew = float(stats.skew(clean))\nkurt = float(stats.kurtosis(clean))  # Fisher excess kurtosis\njb_stat, jb_pvalue = stats.jarque_bera(clean)\nis_normal = bool(jb_pvalue > 0.05)"),
            ("Line 5", "Assemble and return diagnostic dictionary:", "return {'mean': float(clean.mean()), 'is_unbiased': is_unbiased, 'ljung_box_stat': lb_stat, 'ljung_box_pvalue': lb_pvalue, 'is_white_noise': is_white_noise, 'skewness': skew, 'kurtosis': kurt, 'is_normal': is_normal}"),
        ],
        "gotchas": [
            "For Ljung-Box degrees of freedom, dof = lags - K, where K is number of estimated parameters (K=0 for benchmarks, K=p+q for ARIMA).",
            "A model fails if p_value < 0.05 (rejects white noise / normality null).",
        ],
        "verification": "White noise residuals should pass with is_unbiased=True and is_white_noise=True."
    },

    "evaluate_empirical_coverage": {
        "title": "Task 12: Expanding-Window Empirical Coverage Audit (`evaluate_empirical_coverage`)",
        "signature": "def evaluate_empirical_coverage(y_series: pd.Series, method: str, initial_train_size: int, h: int, level: int = 95, m: Optional[int] = None) -> float",
        "objective": "Conducts an expanding walk-forward backtest to verify if a 95% prediction interval truly captures the actual observation 95% of the time.",
        "line_by_line": [
            ("Line 1", "Initialize counters and validate bounds:", "N = len(y_series)\nhits = 0\ntotal = 0"),
            ("Line 2", "Slide origin t from initial_train_size up to N - h:", "for t in range(initial_train_size, N - h + 1):\n    y_tr = y_series.iloc[:t]\n    actual_val = float(y_series.iloc[t + h - 1])"),
            ("Line 3", "Generate point forecast and analytical prediction intervals on y_tr:", "    # Compute point forecast fc and residuals res...\n    pi = compute_prediction_intervals(fc, res, method, levels=[level], m=m)"),
            ("Line 4", "Extract interval bounds for lead time h:", "    lower = float(pi[f'lower_{level}'].iloc[h - 1])\n    upper = float(pi[f'upper_{level}'].iloc[h - 1])"),
            ("Line 5", "Check if actual value falls within interval:", "    if lower <= actual_val <= upper:\n        hits += 1\n    total += 1"),
            ("Line 6", "Return empirical coverage ratio:", "return float(hits / total) if total > 0 else 0.0"),
        ],
        "gotchas": [
            "Target observation is at positional index t + h - 1.",
            "If nominal level is 95%, empirical coverage should fall between 88% and 98% in a reasonably calibrated model.",
        ],
        "verification": "Returns a float in [0.0, 1.0]."
    },

    "forecast_stlf": {
        "title": "Task 13: STL Decomposition Forecasting (`forecast_stlf`)",
        "signature": "def forecast_stlf(y_train: pd.Series, h: int, m: int, method_trend: str = 'drift') -> pd.Series",
        "objective": "Implements Hyndman's STLF method: decomposes y_t = S_t + A_t via STL, forecasts S_t via Seasonal Naive, forecasts A_t via Drift, and recombines.",
        "line_by_line": [
            ("Line 1", "Fit robust STL decomposition with seasonal period m:", "stl = STL(y_train, period=m, robust=True).fit()"),
            ("Line 2", "Extract seasonal component and compute seasonally adjusted series A_t:", "seasonal = stl.seasonal\nseasonally_adj = y_train - seasonal"),
            ("Line 3", "Forecast seasonal component using Seasonal Naive:", "fc_seasonal = forecast_snaive(seasonal, h=h, m=m)"),
            ("Line 4", "Forecast seasonally adjusted series using Drift or Naive:", "if method_trend == 'drift':\n    fc_adj = forecast_drift(seasonally_adj, h=h)\nelif method_trend == 'naive':\n    fc_adj = forecast_naive(seasonally_adj, h=h)"),
            ("Line 5", "Recombine (re-seasonalise) by summing predictions:", "fc_final = fc_seasonal + fc_adj\nfc_final.name = y_train.name\nreturn fc_final"),
        ],
        "gotchas": [
            "Ensure period=m matches the frequency of your data (e.g. m=4 for quarterly beer, m=12 for monthly sales).",
            "Always add the two forecast Series together; pandas will align them by index automatically.",
        ],
        "verification": "STLF captures both seasonal peaks and long-run trend without fitting complex polynomials."
    }
}


def print_banner():
    print(f"\n{BOLD}{CYAN}══════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}  Workshop 03 Hints — The Forecaster's Toolbox (Benchmarks & Diagnostics)     {RESET}")
    print(f"{BOLD}{CYAN}══════════════════════════════════════════════════════════════════════════════{RESET}\n")


def print_task_hint(task_name: str):
    if task_name not in HINTS:
        print(f"{YELLOW}Warning: Task '{task_name}' not found.{RESET}")
        print(f"Available tasks: {', '.join(HINTS.keys())}\n")
        return

    h = HINTS[task_name]
    print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────{RESET}")
    print(f"{BOLD}{GREEN}{h['title']}{RESET}")
    print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────{RESET}")
    print(f"{BOLD}Signature:{RESET}\n  {YELLOW}{h['signature']}{RESET}\n")
    print(f"{BOLD}Objective:{RESET}\n  {h['objective']}\n")
    print(f"{BOLD}Step-by-Step Blueprint (Line-by-Line):{RESET}")

    for item in h["line_by_line"]:
        if len(item) == 3:
            step_num, desc, code = item
            print(f"  {CYAN}▸ {BOLD}{step_num}:{RESET} {desc}")
        elif len(item) == 2:
            desc, code = item
            print(f"  {CYAN}▸ {BOLD}{desc}{RESET}")
        else:
            continue
        if code:
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
    """Prints list of all available tasks in Workshop 03."""
    print_banner()
    print(f"{BOLD}Available Workshop 03 Tasks:{RESET}")
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
    parser = argparse.ArgumentParser(description="Workshop 03 Line-by-Line Hints")
    parser.add_argument("--task", type=str, help="Specific task name (e.g. forecast_snaive, check_residuals)")
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
