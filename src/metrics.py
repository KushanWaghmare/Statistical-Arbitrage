"""
Performance metrics calculation module for statistical arbitrage pairs trading.

Computes institutional-grade metrics: Sharpe ratio, Sortino ratio, Calmar ratio,
maximum drawdown, win rate, profit factor, and trade statistics.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd


def calculate_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """
    Calculate the underwater drawdown series from an equity curve.

    Parameters
    ----------
    equity_series : pd.Series
        Portfolio equity series over time.

    Returns
    -------
    pd.Series
        Drawdown series expressed as a negative percentage.
    """
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return drawdown


def calculate_performance_metrics(
    daily_returns: pd.Series,
    equity_series: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    daily_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive strategy performance metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio return series.
    equity_series : pd.Series
        Cumulative portfolio equity series.
    trades_df : pd.DataFrame
        DataFrame of executed trades.
    risk_free_rate : float
        Annualised risk-free rate (default: 0.02).

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    clean_returns = daily_returns.dropna()
    n_days = len(clean_returns)

    if n_days == 0 or equity_series.iloc[0] == 0:
        return {}

    initial_eq = float(equity_series.iloc[0])
    final_eq = float(equity_series.iloc[-1])
    total_return = (final_eq - initial_eq) / initial_eq

    # Annualised return (CAGR)
    years = n_days / 252.0
    if years > 0 and (1.0 + total_return) > 0:
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Annualised volatility
    daily_vol = float(clean_returns.std())
    annualised_vol = daily_vol * np.sqrt(252.0)

    # Sharpe ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = clean_returns - daily_rf
    if daily_vol > 1e-8:
        sharpe_ratio = float((clean_returns.mean() - daily_rf) / daily_vol * np.sqrt(252.0))
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (downside deviation)
    negative_returns = clean_returns[clean_returns < 0.0]
    downside_std = float(negative_returns.std()) if len(negative_returns) > 1 else 0.0
    if downside_std > 1e-8:
        sortino_ratio = float((clean_returns.mean() - daily_rf) / downside_std * np.sqrt(252.0))
    else:
        sortino_ratio = 0.0

    # Drawdown metrics
    drawdown_series = calculate_drawdown_series(equity_series)
    max_drawdown = float(drawdown_series.min())

    # Calmar ratio
    if abs(max_drawdown) > 1e-6:
        calmar_ratio = float(cagr / abs(max_drawdown))
    else:
        calmar_ratio = 0.0

    # Maximum drawdown duration
    is_underwater = drawdown_series < 0.0
    drawdown_periods = (~is_underwater).cumsum()[is_underwater]
    if not drawdown_periods.empty:
        max_dd_duration = int(drawdown_periods.value_counts().max())
    else:
        max_dd_duration = 0

    # Trade statistics
    total_trades = len(trades_df)
    if total_trades > 0 and "net_pnl" in trades_df.columns:
        winning_trades = trades_df[trades_df["net_pnl"] > 0]
        losing_trades = trades_df[trades_df["net_pnl"] < 0]
        n_wins = len(winning_trades)
        n_losses = len(losing_trades)
        win_rate = (n_wins / total_trades) * 100.0

        gross_profits = winning_trades["net_pnl"].sum()
        gross_losses = abs(losing_trades["net_pnl"].sum())

        if gross_losses > 1e-6:
            profit_factor = float(gross_profits / gross_losses)
        else:
            profit_factor = float(np.inf) if gross_profits > 0 else 0.0

        avg_trade_pnl = float(trades_df["net_pnl"].mean())
        avg_win = float(winning_trades["net_pnl"].mean()) if n_wins > 0 else 0.0
        avg_loss = float(losing_trades["net_pnl"].mean()) if n_losses > 0 else 0.0
        avg_duration = float(trades_df["holding_days"].mean()) if "holding_days" in trades_df.columns else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_duration = 0.0
        n_wins = 0
        n_losses = 0

    metrics_dict = {
        "Initial Capital": round(initial_eq, 2),
        "Final Equity": round(final_eq, 2),
        "Total Return (%)": round(total_return * 100.0, 2),
        "CAGR (%)": round(cagr * 100.0, 2),
        "Annualised Volatility (%)": round(annualised_vol * 100.0, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown * 100.0, 2),
        "Max Drawdown Duration (days)": max_dd_duration,
        "Calmar Ratio": round(calmar_ratio, 2),
        "Total Trades": total_trades,
        "Winning Trades": n_wins,
        "Losing Trades": n_losses,
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != np.inf else "inf",
        "Average Trade PnL": round(avg_trade_pnl, 2),
        "Average Win PnL": round(avg_win, 2),
        "Average Loss PnL": round(avg_loss, 2),
        "Average Holding Period (days)": round(avg_duration, 1)
    }

    if daily_df is not None and "turnover_cost" in daily_df.columns:
        tot_costs = float(daily_df["turnover_cost"].sum())
        tot_borrow = float(daily_df["borrow_cost"].sum()) if "borrow_cost" in daily_df.columns else 0.0
        net_pnl = float(final_eq - initial_eq)
        gross_pnl = net_pnl + tot_costs + tot_borrow
        metrics_dict["Gross PnL ($)"] = round(gross_pnl, 2)
        metrics_dict["Transaction Friction ($)"] = round(tot_costs, 2)
        metrics_dict["Short Borrow Fees ($)"] = round(tot_borrow, 2)
        metrics_dict["Net PnL ($)"] = round(net_pnl, 2)

    return metrics_dict
