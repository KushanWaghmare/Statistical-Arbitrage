"""
Trading signal generation module for statistical arbitrage pairs trading.

Generates entry, exit, and stop-loss signals based on rolling spread z-scores.
"""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd


def generate_trading_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_loss_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Generate threshold-based trading signals and position targets from a z-score series.

    Trading Rules:
    - Long spread (+1): Enter when z <= -entry_threshold (Buy Asset Y, Short Asset X).
    - Short spread (-1): Enter when z >= +entry_threshold (Short Asset Y, Buy Asset X).
    - Mean-reversion exit (0):
        * Exit Long when z >= -exit_threshold (e.g. crosses zero from below).
        * Exit Short when z <= +exit_threshold (e.g. crosses zero from above).
    - Stop-loss exit (0):
        * Exit Long when z <= -stop_loss_threshold (spread divergence).
        * Exit Short when z >= +stop_loss_threshold (spread divergence).

    Parameters
    ----------
    zscore : pd.Series
        Normalised rolling z-score series.
    entry_threshold : float
        Z-score magnitude required to trigger trade entry (default: 2.0).
    exit_threshold : float
        Z-score level triggering position closure (default: 0.0).
    stop_loss_threshold : float
        Z-score magnitude triggering stop-loss liquidation (default: 4.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'zscore': Input z-score
        - 'position': State of position: +1 (Long), -1 (Short), 0 (Flat)
        - 'signal': Event indicator (+1 = Long Entry, -1 = Short Entry, 2 = Mean Exit, -2 = Stop Loss)
    """
    n = len(zscore)
    positions = np.zeros(n, dtype=int)
    signals = np.zeros(n, dtype=int)

    current_pos = 0

    for i in range(n):
        z = zscore.iloc[i]

        if np.isnan(z):
            positions[i] = 0
            continue

        if current_pos == 0:
            # Flat: evaluate entry conditions
            if z <= -entry_threshold:
                current_pos = 1
                signals[i] = 1  # Long spread entry
            elif z >= entry_threshold:
                current_pos = -1
                signals[i] = -1  # Short spread entry
        elif current_pos == 1:
            # Long spread: check stop-loss or profit-taking exit
            if z <= -stop_loss_threshold:
                current_pos = 0
                signals[i] = -2  # Stop loss
            elif z >= -exit_threshold:
                current_pos = 0
                signals[i] = 2  # Mean reversion exit
        elif current_pos == -1:
            # Short spread: check stop-loss or profit-taking exit
            if z >= stop_loss_threshold:
                current_pos = 0
                signals[i] = -2  # Stop loss
            elif z <= exit_threshold:
                current_pos = 0
                signals[i] = 2  # Mean reversion exit

        positions[i] = current_pos

    df_signals = pd.DataFrame({
        "zscore": zscore,
        "position": positions,
        "signal": signals
    }, index=zscore.index)

    return df_signals
