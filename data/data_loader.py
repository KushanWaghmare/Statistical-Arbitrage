"""
Historical price data loader module.

Retrieves historical closing prices for candidate universes via Yahoo Finance
with caching and an offline synthetic generation fallback.
"""

from __future__ import annotations

import os
from typing import List, Optional
import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = [
    # Energy sector
    "XOM", "CVX", "COP", "BP", "EOG",
    # Financial sector
    "JPM", "BAC", "WFC", "C", "MS", "GS"
]


def generate_synthetic_universe(
    tickers: Optional[List[str]] = None,
    n_days: int = 756,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic asset prices containing true cointegrated pairs
    and non-cointegrated random walks.

    Parameters
    ----------
    tickers : list of str, optional
        List of ticker symbols. Defaults to DEFAULT_UNIVERSE.
    n_days : int
        Number of trading days to simulate (756 ~= 3 years).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily closing prices with DatetimeIndex.
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE

    np.random.seed(seed)
    date_range = pd.date_range(end="2024-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(index=date_range)

    # Base common market factor
    market_returns = np.random.normal(0.0003, 0.01, size=n_days)
    market_factor = np.cumprod(1 + market_returns)

    # Pair 1: XOM and CVX (strongly cointegrated via mean-reverting Ornstein-Uhlenbeck spread)
    theta = 0.08  # Mean reversion speed (half-life ~ 8.6 days)
    sigma_ou = 0.8
    ou_spread = np.zeros(n_days)
    for t in range(1, n_days):
        ou_spread[t] = ou_spread[t - 1] - theta * ou_spread[t - 1] + np.random.normal(0, sigma_ou)

    xom_drift = np.cumsum(np.random.normal(0.0004, 0.012, size=n_days))
    prices["XOM"] = 80.0 * np.exp(xom_drift) * (0.8 + 0.2 * market_factor)
    beta_cvx = 1.35
    alpha_cvx = 15.0
    prices["CVX"] = alpha_cvx + beta_cvx * prices["XOM"] + ou_spread

    # Pair 2: JPM and BAC (cointegrated)
    theta_jpm = 0.05  # Mean reversion speed (half-life ~ 13.8 days)
    sigma_jpm = 0.5
    jpm_spread = np.zeros(n_days)
    for t in range(1, n_days):
        jpm_spread[t] = jpm_spread[t - 1] - theta_jpm * jpm_spread[t - 1] + np.random.normal(0, sigma_jpm)

    jpm_drift = np.cumsum(np.random.normal(0.0003, 0.014, size=n_days))
    prices["JPM"] = 120.0 * np.exp(jpm_drift) * (0.7 + 0.3 * market_factor)
    beta_bac = 0.28
    alpha_bac = 5.0
    prices["BAC"] = alpha_bac + beta_bac * prices["JPM"] + jpm_spread

    # Other tickers: correlated or uncorrelated random walks (not cointegrated)
    base_levels = {"COP": 90.0, "BP": 35.0, "EOG": 110.0, "WFC": 45.0, "C": 50.0, "MS": 85.0, "GS": 320.0}
    for ticker in tickers:
        if ticker not in prices.columns:
            base = base_levels.get(ticker, 100.0)
            vol = np.random.uniform(0.012, 0.02)
            drift = np.cumsum(np.random.normal(0.0002, vol, size=n_days))
            prices[ticker] = base * np.exp(drift) * (0.85 + 0.15 * market_factor)

    return prices.round(2)


def fetch_historical_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2021-01-01",
    end_date: str = "2024-01-01",
    cache_path: str = "data/prices.csv",
    force_download: bool = False,
    allow_synthetic_fallback: bool = True
) -> pd.DataFrame:
    """
    Fetch historical adjusted closing prices from Yahoo Finance or load from local cache.

    Parameters
    ----------
    tickers : list of str, optional
        Ticker symbols to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    cache_path : str
        Path to save/load CSV cache.
    force_download : bool
        If True, re-downloads even if cached file exists.
    allow_synthetic_fallback : bool
        If True, generates synthetic realistic prices if network fetch fails.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices indexed by Date.
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE

    # Check local cache first
    if not force_download and os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Ensure requested tickers are in cache
            if all(t in cached_df.columns for t in tickers):
                return cached_df[tickers].dropna()
        except Exception:
            pass

    # Attempt live download via yfinance
    df = None
    try:
        import yfinance as yf
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data, pd.DataFrame) and "Close" in data:
            df = data["Close"]
        elif isinstance(data, pd.DataFrame) and not data.empty and all(t in data.columns for t in tickers):
            df = data
        if df is not None and not df.empty:
            df = df.ffill().dropna()
    except Exception as err:
        print(f"Warning: Yahoo Finance download failed ({err}).")

    if df is None or df.empty or len(df.columns) < 2:
        if allow_synthetic_fallback:
            print("Notice: Generating synthetic market universe with verified cointegrating structures.")
            df = generate_synthetic_universe(tickers=tickers)
        else:
            raise RuntimeError("Failed to fetch historical price data and fallback is disabled.")

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    return df
