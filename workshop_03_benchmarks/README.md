# Workshop 03: The Forecaster\x27s Toolbox — Benchmarks, Residual Forensics & Prediction Intervals

Welcome to **Workshop 03**! This laboratory implements the core forecasting tools and diagnostic instruments from **Lecture 03 (The Forecaster\x27s Toolbox)**, faithfully translating Hyndman & Athanasopoulos *FPP3* Chapter 5 into production-grade Python.

---

## ⏱️ 120-Minute Laboratory Map

| Time | Block | Focus | Core Deliverable |
|---|---|---|---|
| **00:00 – 00:35** (35m) | **Part 1** | **Vectorized Benchmark Engines** | `load_and_index_series`, `forecast_mean`, `forecast_naive`, `forecast_snaive`, `forecast_drift`: Point projections with temporal DatetimeIndex alignment. |
| **00:35 – 00:55** (20m) | **Part 2** | **Fitted Values & Table 5.2 SEs** | `compute_benchmark_residuals`, `compute_forecast_sd`, `compute_prediction_intervals`: Degrees of freedom ($K$) and horizon variance scaling $\sigma\sqrt{h}$. |
| **00:55 – 01:15** (20m) | **Part 3** | **Bootstrapped Residual Simulation** | `bootstrap_prediction_intervals`: Sequential path simulation by sampling historical innovation residuals with replacement ($B=1,000$). |
| **01:15 – 01:40** (25m) | **Part 4** | **Automated Residual Diagnostics** | `check_residuals`: Replicate `gg_tsresiduals` (time plot, ACF with Bartlett bounds $\pm 1.96/\sqrt{T}$, histogram) and Portmanteau Ljung-Box test ($DoF = \ell - K$). |
| **01:40 – 02:00** (20m) | **Part 5 & 6** | **Coverage Audits & STLF Pipeline** | `evaluate_empirical_coverage` (rolling origin backtests) & `forecast_stlf` (decomposition forecasting with seasonal naive + drift). |

---

## 🎯 Learning Objectives

1. **Foundational Benchmark Hurdles (FPP3 §5.2):** Implement vectorized Mean, Naïve, Seasonal Naïve, and Drift methods. Understand why every machine learning model must justify its cost against these baselines.
2. **In-Sample Fit vs Future Forecasts (FPP3 §5.3):** Compute one-step fitted values $\hat{y}_{t|t-1}$ and residuals $e_t$. Audit parameter leakage where in-sample fit cheats using full-sample parameters.
3. **Multi-Step Uncertainty Quantification (FPP3 §5.5):** Derive and implement exact Table 5.2 standard errors. Understand why Naïve variance expands as $\sigma\sqrt{h}$ and Seasonal Naïve variance jumps stepwise at annual boundaries.
4. **Distribution-Free Bootstrapping (FPP3 §5.5):** Simulate ensembles of future sample paths when Gaussian normality breaks down under fat tails and asymmetric economic shocks.
5. **Automated Diagnostic Suite (`check_residuals`):** Build a 4-stage residual audit evaluating zero mean (t-test), white noise (Ljung-Box test with parameter-adjusted DoF), homoscedasticity, and Gaussian normality.
6. **Empirical Coverage Verification:** Execute rolling-origin backtests to verify whether nominal 95% intervals achieve true 95% empirical coverage in real economic time series.
7. **Forecasting with Decomposition (STLF, FPP3 §5.7):** Build an automated STLF pipeline combining STL decomposition with Seasonal Naïve and Drift extrapolation.

---

## 📁 Folder Structure

```
workshop_03_benchmarks/
├── README.md
├── data/
│   ├── RSAFS.csv               # US Monthly Retail Sales (1992 - 2024)
│   ├── INDPRO.csv              # US Industrial Production Index
│   ├── aus_beer.csv            # Australian Quarterly Beer Production (FPP3 Fig 5.7)
│   ├── gafa_stock_google.csv   # Google 2015 Daily Closing Price (FPP3 Fig 5.13)
│   └── annual_egg_prices.csv   # Annual Egg Prices 1900-1993 (FPP3 Fig 5.17)
├── instructor/
│   ├── workshop_03_solution.ipynb
│   └── workshop_03_solution.py
├── student/
│   └── workshop_03.ipynb
└── tests/
    ├── nb_loader.py
    └── test_workshop_03.py
```

---

## 🧪 Testing Locally

To run the autograder against your student notebook:
```bash
pytest workshop_03_benchmarks/tests/ -v
```

To run tests against the instructor solution:
```bash
AUTOGRADE_TARGET=instructor pytest workshop_03_benchmarks/tests/ -v
```

---

## 📊 Grading Rubric (100 Points)

| Test Case | Focus | Points |
|:---|:---|:---:|
| `test_load_and_index_series` | DatetimeIndex enforcement, calendar frequency & metadata preservation | 10 |
| `test_benchmark_forecasting_engines` | Point forecast calculations (Mean, Naïve, SNaïve, Drift) & future index alignment | 20 |
| `test_benchmark_residuals` | One-step in-sample residuals & exact NaN initialization geometry | 15 |
| `test_forecast_standard_deviations_table52` | Table 5.2 multi-step standard errors & horizon scaling factors | 15 |
| `test_prediction_intervals_and_bootstrapping` | Analytical Gaussian bands & empirical bootstrapped quantile envelopes | 15 |
| `test_check_residuals_diagnostics` | Automated diagnostic battery (unbiasedness, Ljung-Box test & DoF) | 15 |
| `test_forecast_stlf` | STLF decomposition pipeline (STL + Seasonal Naïve + Drift) | 10 |
| **Total** | | **100** |
