# Workshop 01: Time Series Graphics & Diagnostics

Welcome to **Workshop 01**! This workshop implements the core concepts from **Lecture 01 (The Forecasting Problem & Time Series Graphics)**, translating Hyndman & Athanasopoulos *FPP3* Chapters 1–2 into production-grade Python.

---

## ⏱️ 120-Minute Laboratory Map

| Time | Block | Focus | Core Deliverable |
|---|---|---|---|
| **00:00 – 00:25** (25m) | **Part 1** | **Temporal Data Ingestion** | `load_and_index_series`: `DatetimeIndex`, monotonic sorting, `asfreq('MS')`. |
| **00:25 – 00:55** (30m) | **Part 2** | **Visual EDA & Seasonality** | `extract_seasonal_components`: Time plot, seasonal plot, seasonal subseries plot. |
| **00:55 – 01:15** (20m) | **Part 3** | **Lag Scatter Grids** | `build_lag_matrix`: Lag scatter grid for lags 1–12. |
| **01:15 – 01:40** (25m) | **Part 4** | **ACF & Ljung-Box Test** | `compute_sample_acf` & `test_white_noise`: Sample ACF, Bartlett bounds, Ljung-Box test. |
| **01:40 – 02:00** (20m) | **Part 5** | **Knowledge Checks & Git** | 3 syllabus practical questions answered in markdown. |

---

## 🎯 Learning Objectives

1. **Temporal Data Integrity:** Enforce `DatetimeIndex` with explicit frequency; distinguish `asfreq` from `resample`.
2. **Three Canonical Plots:** Time plots, seasonal overlay plots, and seasonal subseries plots.
3. **Lag Geometry:** Inspect serial dependence visually via lag scatter matrices.
4. **Autocorrelation & White Noise:** Compute sample ACF $r_k$, Bartlett bounds, and execute the Ljung-Box portmanteau test.
5. **Professional Git Hygiene:** Proper `.gitignore` practices for API keys, data, and environments.

---

## 📁 Folder Structure

```
workshop_01_ts_graphics/
├── README.md
├── data/
│   ├── INDPRO.csv          # US Industrial Production Index
│   ├── RSAFS.csv           # US Retail & Food Services Sales (Unadjusted)
│   └── RSAFS_SA.csv        # US Retail Sales (Seasonally Adjusted)
├── instructor/
│   ├── workshop_01_solution.ipynb
│   └── workshop_01_solution.py
├── student/
│   └── workshop_01.ipynb
└── tests/
    ├── nb_loader.py
    └── test_workshop_01.py
```

---

## 🧪 Testing Locally

```bash
pytest workshop_01_ts_graphics/tests/ -v
```

All 6 test assertions should pass cleanly.

---

## 📊 Grading Rubric (100 Points)

| Criterion | Points |
|:---|:---:|
| **Task 1: Temporal Ingestion & Frequency** | 20 |
| **Task 2: Seasonal Component Extraction** | 20 |
| **Task 3: Lag Matrix Construction** | 20 |
| **Task 4: Sample ACF Computation** | 20 |
| **Task 5: Ljung-Box White Noise Test** | 10 |
| **Task 6: Knowledge Questions** | 10 |
