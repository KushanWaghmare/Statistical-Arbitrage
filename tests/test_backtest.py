"""
Unit tests for backtesting engine and portfolio accounting.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest


def test_run_backtest_execution_lag():
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    py = pd.Series([100.0, 100.0, 105.0, 110.0, 110.0], index=dates)
    px = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0], index=dates)
    beta = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)

    # Signal triggers on day 1 (pos = 1)
    target_pos = pd.Series([0, 1, 1, 0, 0], index=dates)

    res = run_backtest(
        price_y=py,
        price_x=px,
        hedge_ratio=beta,
        positions=target_pos,
        initial_capital=10000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        borrow_rate_bps=0.0
    )

    daily = res["daily_df"]

    # Verify actual position is 0 on day 1 (lagged by 1 day) and 1 on day 2
    assert daily["actual_pos"].iloc[0] == 0
    assert daily["actual_pos"].iloc[1] == 0
    assert daily["actual_pos"].iloc[2] == 1


def test_run_backtest_costs_deducted():
    dates = pd.date_range("2023-01-01", periods=4, freq="B")
    py = pd.Series([100.0, 100.0, 100.0, 100.0], index=dates)
    px = pd.Series([50.0, 50.0, 50.0, 50.0], index=dates)
    beta = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
    target_pos = pd.Series([1, 1, 0, 0], index=dates)

    res = run_backtest(
        price_y=py,
        price_x=px,
        hedge_ratio=beta,
        positions=target_pos,
        initial_capital=10000.0,
        commission_bps=10.0,  # 10 bps
        slippage_bps=10.0,    # 10 bps
        borrow_rate_bps=0.0
    )

    daily = res["daily_df"]
    # Prices did not change, but turnover occurred, so equity must decrease due to costs
    assert res["final_equity"] < 10000.0
    assert daily["turnover_cost"].sum() > 0
