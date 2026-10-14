# Machine Learning II — Applied Workshops (Python)

Welcome to the hands-on workshop repository for **Machine Learning II: Forecasting and Dynamic Modeling** at Lazarski University.

This repository contains all 16 applied computing workshops, accompanying datasets, unit tests, and GitHub Classroom autograding pipelines.

---

## 📘 Student Guide: How to Work and Submit

All workshop assignments are delivered and graded through GitHub Classroom using Python and Jupyter Notebooks.

### 1. Initial Setup (Do this once at the start of the semester)

- **Option A: Conda / Mamba (Recommended)**:
  ```bash
  conda env create -f environment.yml
  conda activate ml2_workshops
  python -m ipykernel install --user --name ml2_workshops --display-name "Python (ML II)"
  ```

- **Option B: Standard virtualenv + pip**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate       # Windows: .venv\Scripts\activate
  pip install -e ".[test]"
  ```

- **Launch Jupyter Lab**:
  ```bash
  jupyter lab
  ```

---

### 2. Weekly Assignment Routine

For each assigned workshop (e.g., **Workshop 00**):

1. **Open the Student Notebook**:
   In Jupyter Lab, open:
   `workshop_00_revising/student/workshop_00.ipynb`
   *(Ensure your Jupyter kernel in the top-right corner is set to `Python (ML II)`).*

2. **Read the Instructions & Write Your Code**:
   Each section contains conceptual explanations and function stubs. Replace `raise NotImplementedError` with your implementation:
   ```python
   def generate_panel_data(n_entities=10, n_periods=24, n_features=6, random_state=42):
       """Generates a synthetic panel dataset with MultiIndex [entity_id, date]."""
       # YOUR CODE HERE
       return df
   ```

3. **Save Your Notebook**:
   Press <kbd>Ctrl+S</kbd> / <kbd>Cmd+S</kbd> to save changes to disk before running tests.

---

### 3. Self-Check & Local Autograding

You can grade yourself locally before pushing to GitHub!

- **In the Terminal**:
  ```bash
  pytest workshop_00_revising/tests/
  ```

- **Directly Inside Jupyter**:
  Run this command in any notebook cell:
  ```python
  !pytest ../tests/ -v
  ```

#### Reading Test Results:
- 🟢 **`PASSED`**: Your implementation satisfies shapes, indices, types, and logic.
- 🔴 **`FAILED - NotImplementedError: Student implementation missing`**: Task not completed yet.
- 🔴 **`FAILED - AssertionError: Data leakage detected!`**: Pedagogical error (e.g. leaking future or cross-entity data). Read the traceback to fix your logic!

---

### 4. Submitting via GitHub Classroom

Once all local tests pass:

```bash
git add workshop_00_revising/student/workshop_00.ipynb
git commit -m "Complete Workshop 00"
git push origin main
```

Check the **Actions** tab on your GitHub repository. The autograder will run and display your official grade.

---

### ⚠️ Common Student Pitfalls & Rules

| ❌ What NOT to do | ✅ What to do instead |
|---|---|
| **Do not change function signatures** (names, parameter names, or defaults). | Keep the `def function_name(...):` signature exactly as provided. |
| **Do not use R packages or `rpy2`.** | Strictly Python. All coursework is pure Python. |
| **Do not forget to save your notebook before running `pytest`.** | The autograder tests the saved `.ipynb` file on disk. |
| **Do not use random shuffles on panel / time series data.** | Always use group-aware (`GroupKFold`) or temporal (`TimeSeriesSplit`) splitting. |

---

## Workshop Schedule & Topic Index

| # | Workshop Folder | Focus & Prerequisite Alignment | Deliverables |
|:---:|:---|:---|:---:|
| **00** | [`workshop_00_revising`](workshop_00_revising/) | **Foundations Bridge**: Synthetic panel data, Pandas refresher, PCA / t-SNE / UMAP, Scikit-Learn Pipelines & GroupKFold | Notebook & Tests |
| **01** | `workshop_01_ts_graphics` | **Lesson 01**: FRED/ECB APIs, datetime indexing, seasonal & lag plots, autocorrelation (ACF) | Notebook & Tests |
| **02** | `workshop_02_decomposition` | **Lesson 02**: Box-Cox, classical & STL decomposition, time series features, leak-free imputation | Notebook & Tests |
| **03** | `workshop_03_benchmarks` | **Lesson 03**: Naive, seasonal naive, drift benchmarks, residual diagnostics, prediction intervals | Notebook & Tests |
| **04** | `workshop_04_evaluation` | **Lesson 04**: MAE, RMSE, MAPE, MASE, walk-forward CV, Diebold–Mariano test | Notebook & Tests |
| **05** | `workshop_05_regression` | **Lesson 05**: Time series regression, trend/seasonality regressors, spurious regression | Notebook & Tests |
| **06** | `workshop_06_ets` | **Lesson 06**: Exponential smoothing, Holt-Winters, automated ETS selection via AICc | Notebook & Tests |
| **07** | `workshop_07_arima` | **Lesson 07**: Stationarity tests (ADF/KPSS), differencing, ACF/PACF identification, ARIMA(p,d,q) | Notebook & Tests |
| **08** | `workshop_08_advanced_arima` | **Lesson 08**: Seasonal ARIMA (SARIMA), SARIMAX with exogenous predictors, Fourier terms | Notebook & Tests |
| **09** | `workshop_09_prophet_garch` | **Lesson 09**: Additive models (Prophet), volatility clustering, ARCH/GARCH modeling | Notebook & Tests |
| **10** | `workshop_10_deep_learning` | **Lesson 10**: Sequence windowing, recurrent architectures (RNN, LSTM, GRU) in PyTorch | Notebook & Tests |
| **11** | `workshop_11_transformers` | **Lesson 11**: Temporal self-attention, Temporal Fusion Transformer (TFT) | Notebook & Tests |
| **12** | `workshop_12_hierarchical` | **Lesson 12**: Hierarchical & grouped time series, bottom-up, top-down, MinT optimal reconciliation | Notebook & Tests |
| **13** | `workshop_13_multivariate` | **Lesson 13**: Vector Autoregressions (VAR), Granger causality, Impulse Response Functions (IRF) | Notebook & Tests |
| **14** | `workshop_14_practical` | **Lesson 14**: End-to-end production pipelines, automated backtesting, monitoring & drift | Notebook & Tests |
| **15** | `workshop_15_final_project` | **Lesson 15**: Capstone forecasting tournament & comprehensive modeling report | Project Submission |

---

## 📊 Workshop Progress & Grade Tracker

<!-- AUTOGRADER_GRADES_START -->

**Total Progress:** `0 / 1600 pts` across 16 workshops.

| # | Workshop | Topic Focus | Grade | Attempts | Status | Last Run |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **00** | [`workshop_00_revising`](workshop_00_revising/) | Foundations: Synthetic Panel, PCA/UMAP, Pipelines | — | 0 | ⚪ Pending | — |
| **01** | [`workshop_01_ts_graphics`](workshop_01_ts_graphics/) | Time Series Graphics: APIs, Datetime Indexing, ACF | — | 0 | ⚪ Pending | — |
| **02** | `workshop_02_decomposition` | Transformations, Classical & STL Decomposition | — | 0 | ⚪ Pending | — |
| **03** | `workshop_03_benchmarks` | Benchmarks, Diagnostics & Prediction Intervals | — | 0 | ⚪ Pending | — |
| **04** | `workshop_04_evaluation` | Forecast Accuracy Metrics & Walk-Forward CV | — | 0 | ⚪ Pending | — |
| **05** | `workshop_05_regression` | Time Series Regression & Spurious Correlation | — | 0 | ⚪ Pending | — |
| **06** | `workshop_06_ets` | Exponential Smoothing & ETS Model Selection | — | 0 | ⚪ Pending | — |
| **07** | `workshop_07_arima` | Stationarity, Differencing & ARIMA Identification | — | 0 | ⚪ Pending | — |
| **08** | `workshop_08_advanced_arima` | Seasonal ARIMA, SARIMAX & Fourier Terms | — | 0 | ⚪ Pending | — |
| **09** | `workshop_09_prophet_garch` | Additive Models (Prophet) & GARCH Volatility | — | 0 | ⚪ Pending | — |
| **10** | `workshop_10_deep_learning` | Sequence Modeling with PyTorch RNN & LSTM | — | 0 | ⚪ Pending | — |
| **11** | `workshop_11_transformers` | Temporal Fusion Transformer & Self-Attention | — | 0 | ⚪ Pending | — |
| **12** | `workshop_12_hierarchical` | Hierarchical Forecasting & MinT Reconciliation | — | 0 | ⚪ Pending | — |
| **13** | `workshop_13_multivariate` | Vector Autoregression (VAR) & Impulse Response | — | 0 | ⚪ Pending | — |
| **14** | `workshop_14_practical` | Production Pipelines & Automated Backtesting | — | 0 | ⚪ Pending | — |
| **15** | `workshop_15_final_project` | Capstone Tournament & Final Submission | — | 0 | ⚪ Pending | — |

<!-- AUTOGRADER_GRADES_END -->

---

## Instructor Commands (`Makefile`)

- `make scaffold`: Strips instructor solutions into blank student walkthrough notebooks.
- `make test`: Runs all workshop test suites across the repository.
- `make test-00`: Runs unit tests for Workshop 00.
- `make clean`: Cleans up temporary caches (`.pytest_cache`, `__pycache__`, `.ipynb_checkpoints`).

---

## Pedagogical Policy
Please review [`RULES.md`](RULES.md) for strict requirements on:
- **Exclusively Python** (Zero R policy).
- **Temporal & Longitudinal Integrity** (Zero data leakage).
- **Reproducibility** (Global random seed pinning).
