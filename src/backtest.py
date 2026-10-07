"""
Backtesting engine for statistical arbitrage pairs trading.

Implements realistic capital allocation, Half-Kelly and volatility parity sizing,
execution latency (no lookahead), transaction fees, slippage, and short borrow cost accounting.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def run_backtest(
    price_y: pd.Series,
    price_x: pd.Series,
    hedge_ratio: pd.Series,
    positions: pd.Series,
    initial_capital: float = 100000.0,
    leverage: float = 1.0,
    sizing_mode: str = "fixed",
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    borrow_rate_bps: float = 100.0
) -> Dict[str, Any]:
    """
    Execute backtest on a cointegrated pair with transaction costs and dynamic sizing.

    Parameters
    ----------
    sizing_mode : str
        'fixed', 'vol_parity', or 'kelly'
    """
    df = pd.DataFrame({
        "price_y": price_y,
        "price_x": price_x,
        "hedge_ratio": hedge_ratio,
        "target_pos": positions
    }).dropna()

    n = len(df)
    dates = df.index

    # Shift positions by 1 day to strictly prevent lookahead bias
    actual_pos = df["target_pos"].shift(1).fillna(0).astype(int).values
    p_y = df["price_y"].values
    p_x = df["price_x"].values
    beta = df["hedge_ratio"].values

    total_cost_rate = (commission_bps + slippage_bps) * 1e-4
    daily_borrow_rate = (borrow_rate_bps * 1e-4) / 252.0

    shares_y = np.zeros(n)
    shares_x = np.zeros(n)
    equity = np.zeros(n)
    daily_pnl = np.zeros(n)
    transaction_costs = np.zeros(n)
    borrow_costs = np.zeros(n)

    current_equity = initial_capital
    trades: List[Dict[str, Any]] = []
    active_trade: Optional[Dict[str, Any]] = None
    trade_id = 0

    # Running win/loss tracker for Kelly sizing
    wins, losses = 1.0, 1.0
    sum_win, sum_loss = 1.0, 1.0

    for t in range(n):
        target = actual_pos[t]
        prev_shares_y = shares_y[t - 1] if t > 0 else 0.0
        prev_shares_x = shares_x[t - 1] if t > 0 else 0.0

        new_shares_y = 0.0
        new_shares_x = 0.0

        if target != 0:
            eff_lev = leverage
            if sizing_mode == "kelly":
                p = wins / (wins + losses)
                b = (sum_win / wins) / max(0.01, sum_loss / losses)
                kelly_f = 0.5 * (p * b - (1.0 - p)) / max(0.01, b)
                eff_lev = float(np.clip(kelly_f, 0.25, 2.0))
            elif sizing_mode == "vol_parity" and t > 20:
                # Scale inversely to recent 20-day spread return vol
                recent_vol = np.std(p_y[t-20:t] - beta[t] * p_x[t-20:t])
                eff_lev = float(np.clip(1.5 / max(0.1, recent_vol), 0.5, 2.0))

            alloc_capital = current_equity * eff_lev
            unit_cost = p_y[t] + abs(beta[t]) * p_x[t]
            if unit_cost > 0:
                units = alloc_capital / unit_cost
                new_shares_y = target * units
                new_shares_x = -target * beta[t] * units

        # Rebalancing turnover and friction
        turnover = abs(new_shares_y - prev_shares_y) * p_y[t] + abs(new_shares_x - prev_shares_x) * p_x[t]
        cost = turnover * total_cost_rate
        transaction_costs[t] = cost

        # Borrow cost on short positions
        short_val = max(0.0, -new_shares_y * p_y[t]) + max(0.0, -new_shares_x * p_x[t])
        borrow_cost = short_val * daily_borrow_rate
        borrow_costs[t] = borrow_cost

        # Price change PnL
        if t > 0:
            gross_pnl = prev_shares_y * (p_y[t] - p_y[t - 1]) + prev_shares_x * (p_x[t] - p_x[t - 1])
        else:
            gross_pnl = 0.0

        net_pnl = gross_pnl - cost - borrow_cost
        daily_pnl[t] = net_pnl
        current_equity += net_pnl  # O(1) running update

        shares_y[t] = new_shares_y
        shares_x[t] = new_shares_x
        equity[t] = current_equity

        # Trade tracking
        prev_target = actual_pos[t - 1] if t > 0 else 0
        if target != 0 and prev_target == 0:
            trade_id += 1
            active_trade = {
                "trade_id": trade_id,
                "entry_date": dates[t],
                "direction": "LONG_SPREAD" if target == 1 else "SHORT_SPREAD",
                "entry_equity": current_equity,
                "cumulative_pnl": 0.0
            }
        elif target == 0 and prev_target != 0 and active_trade is not None:
            active_trade["exit_date"] = dates[t]
            active_trade["net_pnl"] = active_trade["cumulative_pnl"] + net_pnl
            active_trade["return_pct"] = (active_trade["net_pnl"] / active_trade["entry_equity"]) * 100.0
            active_trade["holding_days"] = (dates[t] - active_trade["entry_date"]).days
            trades.append(active_trade)

            # Update Kelly stats
            if active_trade["net_pnl"] > 0:
                wins += 1.0
                sum_win += active_trade["net_pnl"]
            else:
                losses += 1.0
                sum_loss += abs(active_trade["net_pnl"])

            active_trade = None
        elif active_trade is not None:
            active_trade["cumulative_pnl"] += net_pnl

    if active_trade is not None:
        active_trade["exit_date"] = dates[-1]
        active_trade["net_pnl"] = active_trade["cumulative_pnl"]
        active_trade["return_pct"] = (active_trade["net_pnl"] / active_trade["entry_equity"]) * 100.0
        active_trade["holding_days"] = (dates[-1] - active_trade["entry_date"]).days
        trades.append(active_trade)

    daily_returns = pd.Series(equity, index=dates).pct_change().fillna(0.0)

    daily_df = pd.DataFrame({
        "price_y": p_y,
        "price_x": p_x,
        "hedge_ratio": beta,
        "actual_pos": actual_pos,
        "shares_y": shares_y,
        "shares_x": shares_x,
        "turnover_cost": transaction_costs,
        "borrow_cost": borrow_costs,
        "daily_pnl": daily_pnl,
        "equity": equity,
        "daily_return": daily_returns
    }, index=dates)

    return {
        "daily_df": daily_df,
        "trades": pd.DataFrame(trades),
        "initial_capital": initial_capital,
        "final_equity": float(equity[-1]),
        "total_pnl": float(equity[-1] - initial_capital)
    }
