"""
Unit tests for performance metrics and drawdown calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import calculate_performance_metrics, calculate_drawdown_series


def test_calculate_drawdown_series():
    equity = pd.Series([100.0, 120.0, 110.0, 90.0, 130.0])
    dd = calculate_drawdown_series(equity)

    # At 120, dd = 0; at 110, dd = (110 - 120)/120 = -8.33%; at 90, dd = (90 - 120)/120 = -25%
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0
    assert pytest.approx(dd.iloc[2], rel=1e-3) == -10.0 / 120.0
    assert pytest.approx(dd.iloc[3], rel=1e-3) == -30.0 / 120.0
    assert dd.iloc[4] == 0.0


def test_calculate_performance_metrics():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    equity = pd.Series([100000.0, 101000.0, 102000.0, 101500.0, 103000.0,
                        104000.0, 103500.0, 105000.0, 106000.0, 107000.0], index=dates)
    returns = equity.pct_change().fillna(0.0)

    trades = pd.DataFrame([
        {"trade_id": 1, "net_pnl": 5000.0, "holding_days": 4},
        {"trade_id": 2, "net_pnl": 2000.0, "holding_days": 3}
    ])

    m = calculate_performance_metrics(returns, equity, trades)

    assert m["Initial Capital"] == 100000.0
    assert m["Final Equity"] == 107000.0
    assert m["Total Return (%)"] == 7.0
    assert m["Sharpe Ratio"] > 0
    assert m["Total Trades"] == 2
    assert m["Win Rate (%)"] == 100.0
