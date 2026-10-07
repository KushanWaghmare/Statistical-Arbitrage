"""
Spread and hedge ratio construction module.

Provides static OLS, Total Least Squares (TLS), dynamic Kalman Filter estimation,
and rolling normalised z-score computation.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def calculate_ols_spread(
    series_y: pd.Series,
    series_x: pd.Series
) -> Tuple[float, float, pd.Series]:
    """
    Calculate static hedge ratio via Ordinary Least Squares: Y = alpha + beta * X.
    Uses numpy polyfit directly.
    """
    df = pd.concat([series_y, series_x], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    beta, alpha = np.polyfit(x, y, 1)
    spread = pd.Series(y - (alpha + beta * x), index=df.index, name="spread")

    return float(beta), float(alpha), spread


def calculate_tls_spread(
    series_y: pd.Series,
    series_x: pd.Series
) -> Tuple[float, float, pd.Series]:
    """
    Calculate Total Least Squares (Orthogonal Regression) hedge ratio via SVD.
    Accounts for measurement errors in both Asset Y and Asset X.
    """
    df = pd.concat([series_y, series_x], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    # Centre the data
    mean_x, mean_y = np.mean(x), np.mean(y)
    data_centered = np.column_stack([x - mean_x, y - mean_y])

    # Singular Value Decomposition: right singular vector of smallest singular value
    _, _, vt = np.linalg.svd(data_centered)
    normal = vt[-1]  # [n_x, n_y] where n_x * (x - mx) + n_y * (y - my) = 0

    beta = -normal[0] / normal[1]
    alpha = mean_y - beta * mean_x
    spread = pd.Series(y - (alpha + beta * x), index=df.index, name="tls_spread")

    return float(beta), float(alpha), spread


def calculate_kalman_spread(
    series_y: pd.Series,
    series_x: pd.Series,
    delta: float = 1e-4,
    R: float = 1e-3,
    adaptive: bool = True
) -> Dict[str, Any]:
    """
    Calculate dynamic rolling hedge ratio and spread using an Adaptive Kalman Filter.
    """
    df = pd.concat([series_y, series_x], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    index = df.index
    n = len(y)

    theta = np.zeros(2)
    P = np.eye(2) * 1.0

    Vw = delta / (1.0 - delta)
    Q = Vw * np.eye(2)

    betas = np.zeros(n)
    alphas = np.zeros(n)
    spread = np.zeros(n)
    current_R = R

    for t in range(n):
        H = np.array([1.0, x[t]])
        P_pred = P + Q

        error = y[t] - np.dot(H, theta)
        spread[t] = error

        if adaptive and t > 10:
            # Adapt observation noise from rolling innovation error
            current_R = 0.95 * current_R + 0.05 * (error ** 2)

        F = np.dot(H, np.dot(P_pred, H)) + current_R
        K = np.dot(P_pred, H) / F

        theta = theta + K * error
        P = P_pred - np.outer(K, np.dot(H, P_pred))

        alphas[t] = theta[0]
        betas[t] = theta[1]

    return {
        "beta": pd.Series(betas, index=index, name="kalman_beta"),
        "alpha": pd.Series(alphas, index=index, name="kalman_alpha"),
        "spread": pd.Series(spread, index=index, name="kalman_spread"),
        "variance": current_R
    }


def calculate_rolling_zscore(
    spread: pd.Series,
    window: int = 30
) -> pd.DataFrame:
    """
    Compute rolling mean, rolling standard deviation, and normalised z-score for a spread series.
    """
    rolling_mean = spread.rolling(window=window, min_periods=max(5, window // 2)).mean()
    rolling_std = spread.rolling(window=window, min_periods=max(5, window // 2)).std()

    rolling_std_safe = rolling_std.replace(0, np.nan)
    zscore = ((spread - rolling_mean) / rolling_std_safe).fillna(0.0)

    return pd.DataFrame({
        "spread": spread,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "zscore": zscore
    }, index=spread.index)
