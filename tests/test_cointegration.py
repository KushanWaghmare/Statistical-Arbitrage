"""
Unit tests for pair selection, correlation screening, Hurst exponent, and cointegration testing.
"""

import numpy as np
import pandas as pd
import pytest

from src.pair_selection import (
    screen_correlation,
    calculate_half_life,
    calculate_hurst_exponent,
    test_cointegration as run_coint_test,
    test_johansen as run_johansen_test,
    find_cointegrated_pairs
)


@pytest.fixture
def sample_cointegrated_data():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    x = 100.0 + np.cumsum(np.random.normal(0, 1.0, size=n))

    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + np.random.normal(0, 0.5)

    y = 10.0 + 1.8 * x + spread
    z = 50.0 + np.cumsum(np.random.normal(0, 1.2, size=n))

    return pd.DataFrame({"Y": y, "X": x, "Z": z}, index=dates)


def test_screen_correlation(sample_cointegrated_data):
    pairs = screen_correlation(sample_cointegrated_data, threshold=0.7)
    assert len(pairs) >= 1
    pair_names = [(p[0], p[1]) for p in pairs]
    assert ("Y", "X") in pair_names or ("X", "Y") in pair_names


def test_calculate_half_life():
    np.random.seed(42)
    n = 500
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = 0.85 * series[t - 1] + np.random.normal(0, 1.0)
    hl = calculate_half_life(pd.Series(series))
    assert 2.0 <= hl <= 7.0


def test_calculate_hurst_exponent():
    np.random.seed(42)
    n = 600
    # Mean reverting series: Hurst < 0.5
    mr_series = np.zeros(n)
    for t in range(1, n):
        mr_series[t] = 0.7 * mr_series[t - 1] + np.random.normal(0, 1.0)
    h_mr = calculate_hurst_exponent(pd.Series(mr_series))
    assert h_mr < 0.5


def test_engle_granger_cointegration(sample_cointegrated_data):
    res_coint = run_coint_test(sample_cointegrated_data["Y"], sample_cointegrated_data["X"])
    assert res_coint["is_cointegrated"] is True
    assert res_coint["p_value"] < 0.05
    assert 1.6 <= res_coint["beta"] <= 2.0

    res_non_coint = run_coint_test(sample_cointegrated_data["X"], sample_cointegrated_data["Z"])
    assert res_non_coint["p_value"] > res_coint["p_value"]


def test_johansen(sample_cointegrated_data):
    j_res = run_johansen_test(sample_cointegrated_data[["Y", "X"]])
    assert "trace_rank" in j_res
    assert "cointegrating_vectors" in j_res
    assert j_res["trace_rank"] >= 1


def test_find_cointegrated_pairs(sample_cointegrated_data):
    res_df = find_cointegrated_pairs(sample_cointegrated_data, min_corr=0.6, max_pvalue=0.05)
    assert not res_df.empty
    top_pair = res_df.iloc[0]
    assert set([top_pair["Asset_Y"], top_pair["Asset_X"]]) == {"Y", "X"}
