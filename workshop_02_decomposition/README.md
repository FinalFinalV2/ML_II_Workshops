# Workshop 02: Time Series Decomposition & Feature Engineering

Welcome to **Workshop 02**! This workshop implements the core concepts from **Lecture 02 (Time Series Decompositions & Feature Engineering)**, translating Hyndman & Athanasopoulos *FPP3* Chapter 3 into production-grade Python.

---

## ⏱️ 120-Minute Laboratory Map

| Time | Block | Focus | Core Deliverable |
|---|---|---|---|
| **00:00 – 00:20** (20m) | **Part 1** | **Transformations & Adjustments** | `apply_calendar_and_price_adjustments`: Normalize leap years / month lengths & deflate price inflation via FRED CPI. |
| **00:20 – 00:40** (20m) | **Part 2** | **Variance Stabilization (Guerrero)** | `estimate_guerrero_lambda`, `apply_boxcox`, `invert_boxcox_mean`: Guerrero CV optimization & Taylor bias correction. |
| **00:40 – 01:00** (20m) | **Part 3** | **Classical Moving Averages** | `compute_centered_ma`: $2 \times m$-MA filters, symmetric convolution, and endpoint void diagnosis. |
| **01:00 – 01:25** (25m) | **Part 4** | **Multiplicative Log-STL Decomposition** | `decompose_log_stl`: LOESS decomposition on log-scale guaranteeing $Y_t \equiv T_t \times S_t \times R_t$ and seasonal adjustment. |
| **01:25 – 01:40** (15m) | **Part 5** | **FEASTS Feature Engineering** | `extract_feasts_features`: Scalar diagnostic vectors ($F_T$, $F_S$, peak/trough calendar months, remainder ACF). |
| **01:40 – 02:00** (20m) | **Part 6 & 7** | **Production STLF Forecaster & Knowledge Checks** | `audit_table_shift_leakage`, `evaluate_imputation_leakage`, `forecast_stl_decomposition`, and 3 interview questions. |

---

## 🎯 Learning Objectives

1. **Calendar & Economic Normalization:** Eliminate calendar noise (varying month days, leap years) and isolate real consumption volumes via CPI deflation.
2. **Guerrero Box-Cox Optimization:** Implement Guerrero's (1993) algorithm to stabilize level-dependent variance and apply second-order Taylor corrections to avoid Jensen's inequality bias.
3. **Classical Moving Averages & Endpoint Voids:** Construct $2 \times m$-MA filters and understand why boundary missing values make them unusable for live forecasting.
4. **Enhanced STL Usage (Direction 1):** Implement Multiplicative Log-STL decomposition, validating exact machine-precision multiplicative reconstruction ($Y_t = T_t \times S_t \times R_t$) and seasonal adjustment ($A_t = Y_t / S_t$).
5. **FEASTS Feature Architecture:** Compress full macroeconomic trajectories into diagnostic scalar summary features for fleet-wide model routing.
6. **Enhanced STL Usage (Direction 4):** Build a production STLF baseline forecaster (STL + seasonally adjusted drift + seasonal naïve) and rigorously audit lookahead leakage in splits and imputation.
7. **Interview Mastery:** Address advanced industry interview questions on Jensen's inequality, LOESS boundary extrapolation, and lookahead data leakage.

---

## 📁 Folder Structure

```
workshop_02_decomposition/
├── README.md
├── data/
│   ├── CPIAUCSL.csv        # US Consumer Price Index for All Urban Consumers
│   ├── INDPRO.csv          # US Industrial Production Index
│   ├── RSAFS.csv           # US Retail & Food Services Sales (Unadjusted)
│   └── UNRATE.csv          # US Civilian Unemployment Rate
├── instructor/
│   ├── workshop_02_solution.ipynb
│   └── workshop_02_solution.py
├── student/
│   └── workshop_02.ipynb
└── tests/
    ├── nb_loader.py
    └── test_workshop_02.py
```

---

## 🧪 Testing Locally

To run the autograder against your student notebook:
```bash
pytest workshop_02_decomposition/tests/ -v
```

To run tests against the instructor solution:
```bash
AUTOGRADE_TARGET=instructor pytest workshop_02_decomposition/tests/ -v
```

---

## 📊 Grading Rubric (100 Points)

| Test Case | Focus | Points |
|:---|:---|:---:|
| `test_calendar_and_price_adjustments` | Leap year days/month normalization & CPI deflation arithmetic | 10 |
| `test_guerrero_and_boxcox_inversion` | Guerrero CV minimization, inversion roundtrip & bias correction | 15 |
| `test_centered_ma_endpoint_voids` | $2 \times m$ MA convolution, linear trend recovery & endpoint NaNs | 10 |
| `test_log_stl_multiplicative_reconstruction` | Multiplicative Log-STL identity $Y_t \equiv T_t \times S_t \times R_t$ | 15 |
| `test_feasts_feature_extraction` | $F_T$, $F_S$, peak/trough month, and remainder diagnostics | 15 |
| `test_table_shift_leakage_and_causal_imputation` | Temporal causality checks & lookahead leakage demonstration | 15 |
| `test_stl_decomposition_forecaster` | STLF drift + seasonal naive forecast & recombination | 10 |
| `test_interview_knowledge_questions` | Substantive answers to 3 interview questions | 10 |
| **Total** | | **100** |
