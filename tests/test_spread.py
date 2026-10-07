"""
Unit tests for spread calculation, static OLS, Total Least Squares, dynamic Kalman Filter, and rolling z-scores.
"""

import numpy as np
import pandas as pd
import pytest

from src.spread import (
    calculate_ols_spread,
    calculate_tls_spread,
    calculate_kalman_spread,
    calculate_rolling_zscore
)


@pytest.fixture
def linear_pair():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    x = 50.0 + np.cumsum(np.random.normal(0, 1.0, size=n))
    noise = np.random.normal(0, 0.5, size=n)
    y = 5.0 + 2.0 * x + noise
    return pd.Series(y, index=dates), pd.Series(x, index=dates)


def test_calculate_ols_spread(linear_pair):
    y, x = linear_pair
    beta, alpha, spread = calculate_ols_spread(y, x)
    assert 1.90 <= beta <= 2.10
    assert 4.0 <= alpha <= 6.0
    assert len(spread) == len(y)
    assert abs(spread.mean()) < 0.1


def test_calculate_tls_spread(linear_pair):
    y, x = linear_pair
    beta, alpha, spread = calculate_tls_spread(y, x)
    assert 1.85 <= beta <= 2.15
    assert 3.5 <= alpha <= 6.5
    assert len(spread) == len(y)


def test_calculate_kalman_spread(linear_pair):
    y, x = linear_pair
    kf_res = calculate_kalman_spread(y, x, delta=1e-4, R=1e-3)
    assert "beta" in kf_res
    assert "alpha" in kf_res
    assert "spread" in kf_res
    final_beta = kf_res["beta"].iloc[-1]
    assert 1.80 <= final_beta <= 2.20
    assert len(kf_res["spread"]) == len(y)


def test_calculate_rolling_zscore():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    spread = pd.Series(np.random.normal(0, 1.0, size=n), index=dates)
    z_df = calculate_rolling_zscore(spread, window=20)
    assert "zscore" in z_df.columns
    valid_z = z_df["zscore"].iloc[20:]
    assert abs(valid_z.mean()) < 0.5
    assert 0.5 <= valid_z.std() <= 1.5
