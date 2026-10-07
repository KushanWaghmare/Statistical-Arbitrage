"""
Unit tests for trading signals and threshold logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import generate_trading_signals


def test_generate_trading_signals():
    # Construct synthetic z-score scenario:
    # 0 -> drops to -2.5 (enter long) -> rises to 0.1 (exit) -> rises to 2.2 (enter short) -> rises to 4.2 (stop loss)
    z_values = [
        0.0, -1.0, -2.1, -1.5, -0.5, 0.2, 0.5, 1.5, 2.3, 3.0, 4.2, 2.0, 0.0
    ]
    dates = pd.date_range("2023-01-01", periods=len(z_values), freq="B")
    z_series = pd.Series(z_values, index=dates)

    signals_df = generate_trading_signals(
        z_series,
        entry_threshold=2.0,
        exit_threshold=0.0,
        stop_loss_threshold=4.0
    )

    positions = signals_df["position"].tolist()
    events = signals_df["signal"].tolist()

    # Step 2 (-2.1): Long Entry -> pos=1, event=1
    assert positions[2] == 1
    assert events[2] == 1

    # Step 3 (-1.5) & Step 4 (-0.5): Hold Long
    assert positions[3] == 1
    assert positions[4] == 1

    # Step 5 (0.2): Mean exit -> pos=0, event=2
    assert positions[5] == 0
    assert events[5] == 2

    # Step 8 (2.3): Short Entry -> pos=-1, event=-1
    assert positions[8] == -1
    assert events[8] == -1

    # Step 10 (4.2): Stop loss exit -> pos=0, event=-2
    assert positions[10] == 0
    assert events[10] == -2
