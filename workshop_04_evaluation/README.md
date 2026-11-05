# Workshop 04: Evaluating Forecasts & Walk-Forward Validation

Welcome to **Workshop 04**! This laboratory implements the comprehensive forecast evaluation suite and statistical hypothesis testing instruments from **Lecture 04 (Evaluating Forecast Accuracy)**, translating Hyndman & Athanasopoulos *FPP3* Chapter 5 (Sections 5.8–5.10), Diebold & Mariano (1995), and Harvey, Leybourne & Newbold (1997) into production-grade Python.

---

## ⏱️ 120-Minute Laboratory Map

| Time | Block | Focus | Core Deliverable |
|---|---|---|---|
| **00:00 – 00:30** (30m) | **Part 1** | **Core Point Accuracy Metrics** | `load_macro_series`, `compute_mae`, `compute_rmse`, `compute_mape`: Vectorized point error metrics with epsilon protection against zero-division. |
| **00:30 – 00:50** (20m) | **Part 2** | **Scaled Errors & Benchmark Scaling** | `compute_mase`: Mean Absolute Scaled Error using in-sample seasonal ($m>1$) and non-seasonal ($m=1$) Naive denominators. |
| **00:50 – 01:10** (20m) | **Part 3** | **Distributional & Interval Scoring** | `compute_pinball_loss` (asymmetric quantile loss) and `compute_winkler_score` (interval sharpness vs breach penalties). |
| **01:10 – 01:35** (25m) | **Part 4** | **Walk-Forward Validation Engine** | `generate_walk_forward_splits` (expanding and rolling origin index generators) & `evaluate_horizon_degradation` ($h=1\dots 24$). |
| **01:35 – 02:00** (25m) | **Part 5** | **Forensics & Hypothesis Testing** | `demonstrate_kfold_leakage` (proving the K-fold interpolation crime) & `diebold_mariano_test` (Newey-West HAC variance + HLN small-sample correction). |

---

## 🎯 Learning Objectives

1. **Enforce Temporal Boundaries (FPP3 §5.8):** Formulate strict chronological partitions between training and testing data to prevent future information leakage.
2. **Point Forecast Accuracy Forensics:** Calculate and critique scale-dependent (MAE, RMSE), percentage (MAPE), and scaled (MASE) metrics. Understand why MAPE explodes when actual values hit zero.
3. **Scaled Error Invariance (FPP3 §5.8):** Use MASE to compare forecasts across completely different economic units (e.g. beer production vs retail sales) and evaluate models against the simple Naive hurdle ($\text{MASE} < 1$).
4. **Quantile & Interval Loss:** Implement asymmetric pinball loss for quantile regression and the Winkler score for prediction intervals, penalizing narrow intervals that suffer coverage breaches.
5. **Time Series Cross-Validation (FPP3 §5.9):** Build expanding and rolling window walk-forward validation engines in pure Python.
6. **Autocorrelation Leakage Forensics:** Prove computationally why standard random K-fold cross-validation deflates error by 70–90% on persistent economic time series.
7. **Statistical Testing for Predictive Superiority:** Implement the Diebold–Mariano (1995) test, account for multi-step $MA(h-1)$ serial correlation using Newey-West spectral variance, and apply the Harvey-Leybourne-Newbold (1997) finite-sample correction.

---

## 📁 Folder Structure

```
workshop_04_evaluation/
├── README.md
├── data/
│   ├── RSAFS.csv               # US Monthly Retail Sales (1992 - 2024)
│   ├── INDPRO.csv              # US Industrial Production Index
│   ├── aus_beer.csv            # Australian Quarterly Beer Production
│   └── intermittent_sales.csv  # Synthetic demand series with near-zero observations
├── instructor/
│   ├── workshop_04_solution.ipynb
│   └── workshop_04_solution.py
├── student/
│   └── workshop_04.ipynb
└── tests/
    ├── nb_loader.py
    └── test_workshop_04.py
```

---

## 🧪 Testing Locally

To run the autograder against your student notebook:
```bash
pytest workshop_04_evaluation/tests/ -v
```

To run tests against the instructor solution:
```bash
AUTOGRADE_TARGET=instructor pytest workshop_04_evaluation/tests/ -v
```

---

## 📊 Grading Rubric (100 Points)

| Test Case | Focus | Points |
|:---|:---|:---:|
| `test_load_macro_series` | DatetimeIndex enforcement, monthly frequency, metadata preservation | 10 |
| `test_point_accuracy_metrics` | Vectorized MAE, RMSE, and MAPE with zero-division epsilon guard | 15 |
| `test_mase_scaling_invariance` | MASE in-sample Naive denominator, scale invariance & benchmark hurdle | 15 |
| `test_distributional_scoring` | Pinball loss asymmetric V-slopes & Winkler interval score breach penalties | 15 |
| `test_walk_forward_split_engine` | Expanding vs rolling walk-forward split generation & causal index ordering | 15 |
| `test_kfold_leakage_proof` | Empirical proof of K-fold interpolation crime on persistent AR(1) process | 15 |
| `test_diebold_mariano_hac_hln` | Diebold-Mariano test under $H_0$, Newey-West HAC variance & HLN small-sample factor | 15 |
| **Total** | | **100** |
