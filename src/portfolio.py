"""
Multi-pair portfolio and walk-forward validation module for statistical arbitrage.

Implements multi-pair capital allocation and out-of-sample walk-forward validation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.pair_selection import find_cointegrated_pairs, test_cointegration
from src.spread import calculate_kalman_spread, calculate_rolling_zscore, calculate_ols_spread
from src.signals import generate_trading_signals
from src.backtest import run_backtest
from src.metrics import calculate_performance_metrics


def run_multipair_portfolio(
    prices: pd.DataFrame,
    pairs_df: pd.DataFrame,
    max_pairs: int = 3,
    initial_capital: float = 100000.0,
    use_kalman: bool = True,
    zscore_window: int = 30,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_loss_threshold: float = 4.0
) -> Dict[str, Any]:
    """
    Run backtest across multiple cointegrated pairs simultaneously with equal capital allocation.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical prices for the universe.
    pairs_df : pd.DataFrame
        Ranked cointegrated pairs DataFrame.
    max_pairs : int
        Maximum number of top pairs to include in the portfolio.
    initial_capital : float
        Total portfolio starting capital.
    use_kalman : bool
        If True, use dynamic Kalman Filter hedge ratio; else static OLS.
    zscore_window : int
        Lookback window for spread z-score.
    entry_threshold : float
        Z-score entry threshold.
    exit_threshold : float
        Z-score exit threshold.
    stop_loss_threshold : float
        Z-score stop-loss threshold.

    Returns
    -------
    dict
        Aggregated portfolio backtest results, individual pair results, and metrics.
    """
    selected_pairs = pairs_df.head(max_pairs)
    n_pairs = len(selected_pairs)

    if n_pairs == 0:
        raise ValueError("No cointegrated pairs available to construct portfolio.")

    capital_per_pair = initial_capital / n_pairs
    pair_results = []
    combined_equity_df = pd.DataFrame(index=prices.index)

    for idx, row in selected_pairs.iterrows():
        y_tick = row["Asset_Y"]
        x_tick = row["Asset_X"]
        py = prices[y_tick]
        px = prices[x_tick]

        if use_kalman:
            kf_res = calculate_kalman_spread(py, px)
            spread_series = kf_res["spread"]
            beta_series = kf_res["beta"]
        else:
            beta, alpha, spread_series = calculate_ols_spread(py, px)
            beta_series = pd.Series(beta, index=spread_series.index)

        z_df = calculate_rolling_zscore(spread_series, window=zscore_window)
        signals_df = generate_trading_signals(
            z_df["zscore"],
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss_threshold
        )

        backtest_res = run_backtest(
            price_y=py,
            price_x=px,
            hedge_ratio=beta_series,
            positions=signals_df["position"],
            initial_capital=capital_per_pair
        )

        pair_name = f"{y_tick}-{x_tick}"
        pair_metrics = calculate_performance_metrics(
            daily_returns=backtest_res["daily_df"]["daily_return"],
            equity_series=backtest_res["daily_df"]["equity"],
            trades_df=backtest_res["trades"]
        )

        pair_results.append({
            "pair": pair_name,
            "asset_y": y_tick,
            "asset_x": x_tick,
            "metrics": pair_metrics,
            "backtest": backtest_res
        })

        combined_equity_df[pair_name] = backtest_res["daily_df"]["equity"]

    # Fill any missing dates and sum equity across pairs
    combined_equity_df = combined_equity_df.ffill().bfill()
    portfolio_equity = combined_equity_df.sum(axis=1)
    portfolio_returns = portfolio_equity.pct_change().fillna(0.0)

    # Combine trade logs
    all_trades = []
    for res in pair_results:
        trades = res["backtest"]["trades"].copy()
        if not trades.empty:
            trades["pair"] = res["pair"]
            all_trades.append(trades)

    combined_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    portfolio_metrics = calculate_performance_metrics(
        daily_returns=portfolio_returns,
        equity_series=portfolio_equity,
        trades_df=combined_trades_df
    )

    return {
        "portfolio_equity": portfolio_equity,
        "portfolio_returns": portfolio_returns,
        "portfolio_metrics": portfolio_metrics,
        "combined_trades": combined_trades_df,
        "pair_results": pair_results,
        "individual_equities": combined_equity_df
    }


def run_walk_forward_validation(
    prices: pd.DataFrame,
    in_sample_len: int = 252,
    out_of_sample_len: int = 63,
    min_corr: float = 0.70,
    max_pvalue: float = 0.05
) -> Dict[str, Any]:
    """
    Execute walk-forward out-of-sample validation to verify strategy robustness
    and guard against overfitting and structural regime breaks.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.
    in_sample_len : int
        Number of trading days in calibration / in-sample period (default: 252 ~= 1 year).
    out_of_sample_len : int
        Number of trading days in out-of-sample testing window (default: 63 ~= 3 months).
    min_corr : float
        Correlation threshold for in-sample screening.
    max_pvalue : float
        Cointegration p-value threshold.

    Returns
    -------
    dict
        Walk-forward windows summary and stitched out-of-sample return series.
    """
    total_len = len(prices)
    step = out_of_sample_len

    windows = []
    oos_returns_list = []
    current_start = 0

    while current_start + in_sample_len + out_of_sample_len <= total_len:
        is_end = current_start + in_sample_len
        oos_end = is_end + out_of_sample_len

        is_prices = prices.iloc[current_start:is_end]
        oos_prices = prices.iloc[is_end:oos_end]

        # 1. Screen cointegration strictly in-sample
        coint_pairs = find_cointegrated_pairs(is_prices, min_corr=min_corr, max_pvalue=max_pvalue)

        if not coint_pairs.empty:
            top_pair = coint_pairs.iloc[0]
            y_tick = top_pair["Asset_Y"]
            x_tick = top_pair["Asset_X"]

            # Estimate parameters on in-sample data
            py_is = is_prices[y_tick]
            px_is = is_prices[x_tick]
            beta, alpha, _ = calculate_ols_spread(py_is, px_is)

            # Apply out-of-sample
            py_oos = oos_prices[y_tick]
            px_oos = oos_prices[x_tick]
            oos_spread = py_oos - (alpha + beta * px_oos)

            # Rolling z-score on combined to allow smooth lookback
            combined_py = prices[y_tick].iloc[current_start:oos_end]
            combined_px = prices[x_tick].iloc[current_start:oos_end]
            combined_spread = combined_py - (alpha + beta * combined_px)
            z_df = calculate_rolling_zscore(combined_spread, window=30)
            oos_z = z_df["zscore"].iloc[in_sample_len:]

            signals_df = generate_trading_signals(oos_z)
            backtest_res = run_backtest(
                price_y=py_oos,
                price_x=px_oos,
                hedge_ratio=pd.Series(beta, index=py_oos.index),
                positions=signals_df["position"],
                initial_capital=100000.0
            )

            oos_ret = backtest_res["daily_df"]["daily_return"]
            oos_returns_list.append(oos_ret)

            windows.append({
                "window": len(windows) + 1,
                "in_sample_start": is_prices.index[0].strftime("%Y-%m-%d"),
                "in_sample_end": is_prices.index[-1].strftime("%Y-%m-%d"),
                "out_of_sample_start": oos_prices.index[0].strftime("%Y-%m-%d"),
                "out_of_sample_end": oos_prices.index[-1].strftime("%Y-%m-%d"),
                "selected_pair": f"{y_tick}-{x_tick}",
                "in_sample_pvalue": top_pair["Coint_PValue"],
                "oos_total_return": round((backtest_res["final_equity"] / 100000.0 - 1.0) * 100.0, 2)
            })
        else:
            # No cointegrated pair passed in-sample screen; stay in cash
            dates_oos = oos_prices.index
            zero_ret = pd.Series(0.0, index=dates_oos)
            oos_returns_list.append(zero_ret)

            windows.append({
                "window": len(windows) + 1,
                "in_sample_start": is_prices.index[0].strftime("%Y-%m-%d"),
                "in_sample_end": is_prices.index[-1].strftime("%Y-%m-%d"),
                "out_of_sample_start": oos_prices.index[0].strftime("%Y-%m-%d"),
                "out_of_sample_end": oos_prices.index[-1].strftime("%Y-%m-%d"),
                "selected_pair": "None (Cash)",
                "in_sample_pvalue": np.nan,
                "oos_total_return": 0.0
            })

        current_start += step

    if oos_returns_list:
        stitched_returns = pd.concat(oos_returns_list)
        stitched_equity = (1.0 + stitched_returns).cumprod() * 100000.0
        stitched_metrics = calculate_performance_metrics(
            daily_returns=stitched_returns,
            equity_series=stitched_equity,
            trades_df=pd.DataFrame()
        )
    else:
        stitched_returns = pd.Series()
        stitched_equity = pd.Series()
        stitched_metrics = {}

    return {
        "windows": pd.DataFrame(windows),
        "stitched_returns": stitched_returns,
        "stitched_equity": stitched_equity,
        "stitched_metrics": stitched_metrics
    }
