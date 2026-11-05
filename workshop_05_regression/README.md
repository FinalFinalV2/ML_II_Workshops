# Workshop 05: Regression, Lagged Predictors & Forecast Combinations

Welcome to **Workshop 05** of *Machine Learning II: Forecasting Economic Time Series* at Lazarski University.

In this workshop, you will move beyond univariate statistical benchmarks into **multivariate time series modeling, dynamic lag alignment, and forecast ensembling**.

---

## 🎯 Learning Objectives

By the end of this workshop, you will be able to:

1. **Engineer Secular Trends & Fourier Harmonics:** Build linear and piecewise spline trends, generate Fourier harmonic sine/cosine pairs for seasonal data, and select the optimal harmonic order $K$ using finite-sample $AIC_c$.
2. **Construct Leak-Free Dynamic Lag Matrices:** Formulate dynamic feature tables using `pd.Series.shift()`, mathematically proving why positive shifts (`shift(k)`) are causal while negative shifts (`shift(-k)`) cause catastrophic temporal look-ahead leakage.
3. **Audit Residual Diagnostics:** Perform Durbin-Watson and Breusch-Godfrey Lagrange multiplier tests to detect autocorrelated regression residuals, proving why serially correlated errors invalidate standard OLS inference.
4. **Implement Forecast Combinations & Solve the Combination Puzzle:** Build equal-weight ($1/M$) and Bates-Granger inverse-variance combinations under strict walk-forward cross-validation, verifying empirically why simple averaging defeats estimated optimal weights.

---

## 📁 Repository Structure

```
workshop_05_regression/
├── README.md                          <- This briefing document
├── data/
│   ├── CPIAUCSL.csv                   <- US Headline Consumer Price Index (Monthly)
│   ├── MCOILWTICO.csv                 <- WTI Crude Oil Spot Price (Monthly)
│   ├── RSAFS.csv                      <- US Retail and Food Services Sales (Monthly)
│   └── UNRATE.csv                     <- US Unemployment Rate (Monthly)
├── student/
│   └── workshop_05.ipynb              <- Student assignment notebook to complete
└── tests/
    ├── nb_loader.py                   <- Notebook execution & module extractor
    └── test_workshop_05.py            <- Automated autograding test suite (100 pts)
```

---

## 🧪 Autograding & Local Testing

You can run the test suite locally in your terminal at any time:

```bash
# Run tests from the workshop root:
pytest workshop_05_regression/tests/test_workshop_05.py -v

# Or via Makefile:
make test-w05
```

### Test Breakdown (100 Points Total)
- **Test 1 (10 pts):** `test_01_load_macro_data` — Datetime index monotonicity, frequencies, and null handling.
- **Test 2 (15 pts):** `test_02_trend_and_fourier_features` — Trend splines, Fourier terms, and $AIC_c$ calculation.
- **Test 3 (20 pts):** `test_03_create_lagged_matrix_and_audit` — Causal lag matrix construction and leak-audit assertion.
- **Test 4 (15 pts):** `test_04_residual_diagnostics_dw_bg` — Durbin-Watson statistic and Breusch-Godfrey LM test.
- **Test 5 (15 pts):** `test_05_bates_granger_combination_weights` — Equal-weight and inverse-variance combination schemes.
- **Test 6 (15 pts):** `test_06_walk_forward_ensemble_evaluation` — Walk-forward validation without look-ahead, combination puzzle proof.
- **Test 7 (10 pts):** `test_07_answers_to_practical_questions` — Comprehensive answers to the 3 verbatim syllabus questions.

---

## ⚖️ Rules & Policies
- **Strict Python Only:** Zero R code or packages (`fable`, `tsibble`). Everything is implemented in `pandas`, `numpy`, `scipy`, `statsmodels`, `sklearn`.
- **Zero Information Leakage:** Combination weights must be estimated strictly on training folds; never use the holdout test set to compute combination weights.
