# =============================================================================
# Autograding Tests for Workshop 01: Time Series Graphics & Diagnostics
# =============================================================================

import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from nb_loader import load_submission

module = load_submission()

load_and_index_series = getattr(module, "load_and_index_series", None)
extract_seasonal_components = getattr(module, "extract_seasonal_components", None)
build_lag_matrix = getattr(module, "build_lag_matrix", None)
compute_sample_acf = getattr(module, "compute_sample_acf", None)
func_test_white_noise = getattr(module, "test_white_noise", None)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_and_index_series_frequency():
    """Verifies DatetimeIndex, monotonic sorting, and explicit frequency."""
    assert callable(load_and_index_series), "load_and_index_series not defined"
    s = load_and_index_series(str(DATA_DIR / "INDPRO.csv"), "observation_date", "INDPRO", "MS")

    assert isinstance(s, pd.Series), "Must return pd.Series"
    assert isinstance(s.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert s.index.is_monotonic_increasing, "Index must be monotonically sorted"
    assert s.index.freq is not None, "Series must have explicit frequency"
    assert s.index.freqstr == "MS", f"Expected freq 'MS', got '{s.index.freqstr}'"
    assert s.dtype in (np.float64, np.float32, float), "Values must be float"
    assert len(s) > 100, "Expected a long series"


def test_extract_seasonal_components_shape():
    """Verifies Year x Month matrix structure."""
    assert callable(extract_seasonal_components), "extract_seasonal_components not defined"
    s = load_and_index_series(str(DATA_DIR / "RSAFS.csv"), "observation_date", "RSAFSNA", "MS")
    table = extract_seasonal_components(s)

    assert isinstance(table, pd.DataFrame), "Must return pd.DataFrame"
    assert list(table.columns) == list(range(1, 13)), f"Columns must be months 1..12, got {list(table.columns)}"
    assert table.index.name == "year", "Index must be named 'year'"
    n_years = s.index.year.nunique()
    assert len(table) == n_years, f"Expected {n_years} rows (years), got {len(table)}"


def test_build_lag_matrix_structure():
    """Verifies column naming and temporal alignment."""
    assert callable(build_lag_matrix), "build_lag_matrix not defined"
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    s = pd.Series(np.arange(24, dtype=float), index=idx, name="test")
    lags = [1, 3, 12]
    result = build_lag_matrix(s, lags)

    assert isinstance(result, pd.DataFrame), "Must return pd.DataFrame"
    assert "y_t" in result.columns, "Must contain y_t column"
    for k in lags:
        col = f"lag_{k}"
        assert col in result.columns, f"Must contain {col} column"
        assert pd.isna(result[col].iloc[k - 1]), f"First {k} values of {col} must be NaN"
        # Verify correct shift: lag_k at position k should equal y_t at position 0
        assert result[col].iloc[k] == result["y_t"].iloc[0], f"Temporal alignment error in {col}"


def test_compute_sample_acf_values():
    """Tests exact ACF formula on deterministic signal."""
    assert callable(compute_sample_acf), "compute_sample_acf not defined"
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    y = pd.Series(np.random.RandomState(42).randn(200), index=idx)

    result = compute_sample_acf(y, max_lag=10)

    assert "acf" in result and "lags" in result
    assert result["acf"][0] == pytest.approx(1.0, abs=1e-10), "r_0 must be 1.0"
    assert result["n_obs"] == 200
    assert len(result["acf"]) == 11
    assert result["bartlett_upper"] == pytest.approx(1.96 / np.sqrt(200), rel=1e-4)
    assert result["bartlett_lower"] == pytest.approx(-1.96 / np.sqrt(200), rel=1e-4)


def test_white_noise_decision_rule():
    """White noise should be identified as such; trended data should not."""
    assert callable(func_test_white_noise), "test_white_noise not defined"

    np.random.seed(42)
    wn = pd.Series(np.random.randn(300), index=pd.date_range("2000-01-01", periods=300, freq="MS"))
    wn_result = func_test_white_noise(wn, max_lag=12)
    assert wn_result["is_white_noise"] is True, f"Simulated WN must have p > 0.05, got p={wn_result['p_value']:.4f}"

    indpro = load_and_index_series(str(DATA_DIR / "INDPRO.csv"), "observation_date", "INDPRO", "MS")
    indpro_result = func_test_white_noise(indpro.dropna(), max_lag=12)
    assert indpro_result["is_white_noise"] is False, f"INDPRO must reject WN, got p={indpro_result['p_value']:.4f}"


def test_interview_knowledge_questions():
    """Verifies that the 3 syllabus interview questions are answered."""
    target = os.environ.get("AUTOGRADE_TARGET", "auto").lower()
    if target in ("instructor", "solution"):
        assert True
        return

    student_nb_path = Path(__file__).resolve().parent.parent / "student" / "workshop_01.ipynb"
    if not student_nb_path.exists():
        pytest.skip("Student notebook not found for interview questions check")

    with open(student_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    questions_found = 0
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if any(q in src for q in ["### Question 1", "### Question 2", "### Question 3"]):
            if i + 1 < len(cells):
                ans_cell = cells[i + 1]
                ans_src = "".join(ans_cell.get("source", [])).strip()
                if len(ans_src) > 30 and "*Write your answer here in markdown...*" not in ans_src:
                    questions_found += 1

    assert questions_found >= 3, (
        f"Please answer all 3 interview questions in markdown cells (found {questions_found}/3 substantive answers)."
    )
