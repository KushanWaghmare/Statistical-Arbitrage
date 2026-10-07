"""
Visualisation module for statistical arbitrage pairs trading.

Generates publication-quality charts for asset prices, cointegration spreads,
dynamic hedge ratios, trading signals, equity curves, and underwater drawdowns.
"""

from __future__ import annotations

import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Set visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_normalised_prices(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    save_path: str = "plots/normalised_prices.png"
) -> None:
    """
    Plot normalised price series for two candidate assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.
    ticker_y : str
        Ticker of Asset Y.
    ticker_x : str
        Ticker of Asset X.
    save_path : str
        Filepath to save the rendered PNG image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    norm_y = (prices[ticker_y] / prices[ticker_y].iloc[0]) * 100.0
    norm_x = (prices[ticker_x] / prices[ticker_x].iloc[0]) * 100.0

    ax.plot(prices.index, norm_y, label=f"{ticker_y} (Normalised)", color="#1f77b4", linewidth=1.5)
    ax.plot(prices.index, norm_x, label=f"{ticker_x} (Normalised)", color="#ff7f0e", linewidth=1.5)

    ax.set_title(f"Normalised Historical Price Co-movement: {ticker_y} vs {ticker_x}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalised Price (Base = 100)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_spread_and_hedge_ratio(
    dates: pd.DatetimeIndex,
    spread_ols: pd.Series,
    spread_kalman: pd.Series,
    beta_kalman: pd.Series,
    beta_ols: float,
    ticker_y: str,
    ticker_x: str,
    save_path: str = "plots/spread_and_hedge_ratio.png"
) -> None:
    """
    Plot comparison between static OLS and dynamic Kalman Filter spread and hedge ratio.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index.
    spread_ols : pd.Series
        Static OLS residual spread.
    spread_kalman : pd.Series
        Dynamic Kalman Filter residual spread.
    beta_kalman : pd.Series
        Time-varying Kalman Filter beta.
    beta_ols : float
        Fixed OLS beta.
    ticker_y : str
        Ticker of Asset Y.
    ticker_x : str
        Ticker of Asset X.
    save_path : str
        Filepath to save image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=300)

    # 1. Spread comparison
    ax1.plot(dates, spread_ols, label=f"Static OLS Spread (Beta = {beta_ols:.3f})", color="#7f7f7f", linestyle="--", alpha=0.8)
    ax1.plot(dates, spread_kalman, label="Dynamic Kalman Filter Spread", color="#2ca02c", linewidth=1.4)
    ax1.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax1.set_title(f"Spread Dynamics: {ticker_y} - Beta * {ticker_x}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Spread Value", fontsize=10)
    ax1.legend(frameon=True, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Hedge ratio comparison
    ax2.plot(dates, beta_kalman, label="Kalman Filter Dynamic Hedge Ratio (Beta_t)", color="#d62728", linewidth=1.6)
    ax2.axhline(beta_ols, color="#1f77b4", linestyle="--", linewidth=1.5, label=f"Static OLS Beta ({beta_ols:.3f})")
    ax2.set_title(f"Hedge Ratio Adaptation: {ticker_y} on {ticker_x}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Hedge Ratio (Beta)", fontsize=10)
    ax2.legend(frameon=True, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_zscore_and_signals(
    zscore_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_loss_threshold: float = 4.0,
    pair_label: str = "Pair",
    save_path: str = "plots/zscore_signals.png"
) -> None:
    """
    Plot normalised z-score with trigger boundaries and executed trade entry/exit markers.

    Parameters
    ----------
    zscore_df : pd.DataFrame
        DataFrame containing 'zscore'.
    signals_df : pd.DataFrame
        DataFrame containing 'signal' and 'position'.
    entry_threshold : float
        Z-score entry threshold.
    exit_threshold : float
        Z-score exit threshold.
    stop_loss_threshold : float
        Z-score stop-loss threshold.
    pair_label : str
        Label for the pair (e.g. 'XOM-CVX').
    save_path : str
        Filepath to save image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)

    z = zscore_df["zscore"]
    dates = z.index

    # Plot z-score curve
    ax.plot(dates, z, label="Spread Z-Score", color="#2b5c8f", linewidth=1.2)

    # Threshold horizontal reference lines
    ax.axhline(entry_threshold, color="#d9534f", linestyle="--", linewidth=1.2, label=f"Short Entry (+{entry_threshold}σ)")
    ax.axhline(-entry_threshold, color="#5cb85c", linestyle="--", linewidth=1.2, label=f"Long Entry (-{entry_threshold}σ)")
    ax.axhline(exit_threshold, color="#6c757d", linestyle=":", linewidth=1.0, label="Mean Exit (0.0σ)")
    ax.axhline(stop_loss_threshold, color="#800080", linestyle="-.", linewidth=1.0, label=f"Stop-Loss (+{stop_loss_threshold}σ)")
    ax.axhline(-stop_loss_threshold, color="#800080", linestyle="-.", linewidth=1.0, label=f"Stop-Loss (-{stop_loss_threshold}σ)")

    # Highlight trading regions
    ax.fill_between(dates, entry_threshold, stop_loss_threshold, color="#d9534f", alpha=0.08)
    ax.fill_between(dates, -entry_threshold, -stop_loss_threshold, color="#5cb85c", alpha=0.08)

    # Overlay signal markers
    sig = signals_df["signal"]
    long_entries = dates[sig == 1]
    short_entries = dates[sig == -1]
    exits = dates[sig == 2]
    stop_losses = dates[sig == -2]

    if len(long_entries) > 0:
        ax.scatter(long_entries, z.loc[long_entries], marker="^", color="#28a745", s=70, zorder=5, label="Long Entry Trigger")
    if len(short_entries) > 0:
        ax.scatter(short_entries, z.loc[short_entries], marker="v", color="#dc3545", s=70, zorder=5, label="Short Entry Trigger")
    if len(exits) > 0:
        ax.scatter(exits, z.loc[exits], marker="o", color="#007bff", s=50, zorder=5, label="Mean Reversion Exit")
    if len(stop_losses) > 0:
        ax.scatter(stop_losses, z.loc[stop_losses], marker="X", color="#6f42c1", s=70, zorder=5, label="Stop-Loss Exit")

    ax.set_title(f"Statistical Arbitrage Signals & Normalised Z-Score ({pair_label})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Spread Z-Score", fontsize=11)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_equity_curve_and_drawdown(
    daily_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    benchmark_equity: Optional[pd.Series] = None,
    save_path: str = "plots/equity_curve.png"
) -> None:
    """
    Plot strategy equity curve and underwater drawdown profile.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily backtest results dataframe containing 'equity'.
    initial_capital : float
        Initial capital amount.
    benchmark_equity : pd.Series, optional
        Optional benchmark equity series for comparison.
    save_path : str
        Filepath to save image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}, dpi=300)

    dates = daily_df.index
    equity = daily_df["equity"]

    # 1. Equity curve
    ax1.plot(dates, equity, label="Pairs Trading Strategy Equity", color="#0b6623", linewidth=1.8)
    if benchmark_equity is not None:
        ax1.plot(dates, benchmark_equity, label="Benchmark (Buy & Hold)", color="#6c757d", linestyle="--", linewidth=1.4)
    ax1.axhline(initial_capital, color="black", linestyle=":", alpha=0.5)

    final_val = equity.iloc[-1]
    total_ret = ((final_val / initial_capital) - 1.0) * 100.0
    ax1.set_title(f"Portfolio Equity Progression (Final Return: {total_ret:+.2f}%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax1.legend(frameon=True, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Underwater drawdown
    peak = equity.cummax()
    drawdown = ((equity - peak) / peak) * 100.0

    ax2.fill_between(dates, drawdown, 0, color="#d9534f", alpha=0.35)
    ax2.plot(dates, drawdown, color="#c9302c", linewidth=1.2)
    ax2.set_title("Underwater Drawdown Profile (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.yaxis.set_major_formatter("{x:.1f}%")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
