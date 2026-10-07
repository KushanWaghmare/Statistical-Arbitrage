"""
Pair selection module for statistical arbitrage.

Implements correlation screening, Engle-Granger two-step cointegration testing,
Johansen multivariate cointegration, Hurst exponent, and Ornstein-Uhlenbeck half-life estimation.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore", category=FutureWarning)


def screen_correlation(
    prices: pd.DataFrame,
    threshold: float = 0.75
) -> List[Tuple[str, str, float]]:
    """Screen candidate universe for pairs exhibiting correlation above a defined threshold."""
    corr = prices.corr()
    return [
        (a, b, float(corr.loc[a, b]))
        for a, b in itertools.combinations(prices.columns, 2)
        if abs(corr.loc[a, b]) >= threshold
    ]


def calculate_hurst_exponent(series: pd.Series, max_lags: int = 40) -> float:
    """
    Calculate Hurst exponent (H) using rescaled range (R/S) analysis.
    H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending.
    """
    vals = series.dropna().values
    n = len(vals)
    if n < max_lags * 2:
        return 0.5

    lags = np.unique(np.logspace(np.log10(5), np.log10(min(max_lags, n // 2)), num=10).astype(int))
    tau = []

    for lag in lags:
        # Standard deviation of differences across lag
        diff = vals[lag:] - vals[:-lag]
        tau.append(np.std(diff))

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0])


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion using an Ornstein-Uhlenbeck process.
    Delta(S_t) = gamma * S_{t-1} + alpha + epsilon_t; Half-life = -ln(2) / gamma.
    """
    clean = spread.dropna()
    if len(clean) < 10:
        return float(np.inf)

    s_lag = clean.shift(1).iloc[1:].values
    s_diff = (clean - clean.shift(1)).iloc[1:].values

    # Bivariate regression: s_diff = gamma * s_lag + c
    gamma, _ = np.polyfit(s_lag, s_diff, 1)

    if gamma < 0:
        return float(-np.log(2.0) / gamma)
    return float(np.inf)


def test_cointegration(
    series_y: pd.Series,
    series_x: pd.Series
) -> Dict[str, Any]:
    """
    Perform Engle-Granger two-step cointegration test between two price series.
    """
    df = pd.concat([series_y, series_x], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    # Engle-Granger cointegration test
    score, p_value, crit_values = coint(y, x)

    # OLS parameter estimation via polyfit
    beta, alpha = np.polyfit(x, y, 1)
    residuals = y - (alpha + beta * x)
    res_series = pd.Series(residuals, index=df.index)

    half_life = calculate_half_life(res_series)
    hurst = calculate_hurst_exponent(res_series)

    return {
        "coint_t": float(score),
        "p_value": float(p_value),
        "crit_1pct": float(crit_values[0]),
        "crit_5pct": float(crit_values[1]),
        "crit_10pct": float(crit_values[2]),
        "beta": float(beta),
        "alpha": float(alpha),
        "half_life": float(half_life),
        "hurst": float(hurst),
        "is_cointegrated": bool(p_value < 0.05 and half_life > 0 and half_life < np.inf and hurst < 0.5)
    }


def test_johansen(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1
) -> Dict[str, Any]:
    """
    Perform Johansen cointegration test for multivariate asset baskets.
    """
    clean_df = df.dropna()
    res = coint_johansen(clean_df, det_order=det_order, k_ar_diff=k_ar_diff)

    # Rank: number of cointegrating vectors at 95% confidence
    trace_r = int(np.sum(res.lr1 > res.cvt[:, 1]))
    eigen_r = int(np.sum(res.lr2 > res.cvm[:, 1]))

    return {
        "trace_stat": res.lr1.tolist(),
        "trace_crit_95": res.cvt[:, 1].tolist(),
        "eigen_stat": res.lr2.tolist(),
        "eigen_crit_95": res.cvm[:, 1].tolist(),
        "cointegrating_vectors": res.evec.tolist(),
        "trace_rank": trace_r,
        "eigen_rank": eigen_r
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    min_corr: float = 0.70,
    max_pvalue: float = 0.05,
    max_half_life: float = 60.0,
    max_hurst: float = 0.50
) -> pd.DataFrame:
    """Screen candidate universe and identify statistically robust cointegrated pairs."""
    candidate_pairs = screen_correlation(prices, threshold=min_corr)
    results = []

    for ticker_a, ticker_b, corr in candidate_pairs:
        test_ab = test_cointegration(prices[ticker_a], prices[ticker_b])
        test_ba = test_cointegration(prices[ticker_b], prices[ticker_a])

        best_test = test_ab if test_ab["p_value"] <= test_ba["p_value"] else test_ba
        y_tick = ticker_a if test_ab["p_value"] <= test_ba["p_value"] else ticker_b
        x_tick = ticker_b if test_ab["p_value"] <= test_ba["p_value"] else ticker_a

        if (best_test["p_value"] <= max_pvalue and
                best_test["half_life"] <= max_half_life and
                best_test["hurst"] <= max_hurst):
            results.append({
                "Asset_Y": y_tick,
                "Asset_X": x_tick,
                "Correlation": round(corr, 4),
                "Coint_PValue": round(best_test["p_value"], 5),
                "Coint_TStat": round(best_test["coint_t"], 4),
                "Hurst": round(best_test["hurst"], 3),
                "Half_Life_Days": round(best_test["half_life"], 1),
                "Hedge_Ratio": round(best_test["beta"], 4),
                "Intercept": round(best_test["alpha"], 4)
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["Coint_PValue", "Half_Life_Days"]).reset_index(drop=True)
    return results_df
