"""
Main execution script for Statistical Arbitrage Pairs Trading.

Orchestrates data ingestion, cointegration pair screening, Johansen testing,
dynamic Kalman Filter and Total Least Squares (TLS) spread construction,
Half-Kelly position sizing, backtesting, metrics reporting, and interactive dashboard generation.
"""

from __future__ import annotations

import argparse
import os
import sys

# Configure local writable directory for Matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".matplotlib")

import numpy as np
import pandas as pd

from data.data_loader import fetch_historical_prices, DEFAULT_UNIVERSE
from src.pair_selection import find_cointegrated_pairs, test_johansen
from src.spread import (
    calculate_ols_spread,
    calculate_tls_spread,
    calculate_kalman_spread,
    calculate_rolling_zscore
)
from src.signals import generate_trading_signals
from src.backtest import run_backtest
from src.metrics import calculate_performance_metrics
from src.visualisation import (
    plot_normalised_prices,
    plot_spread_and_hedge_ratio,
    plot_zscore_and_signals,
    plot_equity_curve_and_drawdown
)
from src.portfolio import run_multipair_portfolio, run_walk_forward_validation
from src.dashboard import generate_html_dashboard


def print_banner(title: str) -> None:
    separator = "=" * 78
    print(f"\n{separator}\n {title.upper()}\n{separator}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Arbitrage Pairs Trading Engine")
    parser.add_argument("--sizing", choices=["fixed", "vol_parity", "kelly"], default="fixed",
                        help="Capital allocation and position sizing model (default: fixed)")
    parser.add_argument("--hedge", choices=["kalman", "tls", "ols"], default="kalman",
                        help="Hedge ratio model for backtesting (default: kalman)")
    args = parser.parse_args()

    print_banner("Statistical Arbitrage Pairs Trading Engine")
    print(f"Configuration: Hedge Model = {args.hedge.upper()} | Sizing = {args.sizing.upper()}")

    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Pull historical price series
    prices = fetch_historical_prices(
        tickers=DEFAULT_UNIVERSE,
        start_date="2021-01-01",
        end_date="2024-01-01",
        cache_path="data/prices.csv"
    )

    print(f"Data ingested successfully: {len(prices)} trading days across {len(prices.columns)} assets.")
    print(f"Universe: {', '.join(prices.columns)}")

    # 2. Pair Screening: Correlation + Engle-Granger Cointegration + Hurst Exponent
    print_banner("Step 1: Cointegration Screening & Statistical Testing")
    print("Testing candidate pairs for Pearson correlation (r >= 0.70) and Engle-Granger cointegration (p < 0.05)...")

    coint_df = find_cointegrated_pairs(prices, min_corr=0.70, max_pvalue=0.05, max_half_life=60.0)
    if coint_df.empty:
        coint_df = find_cointegrated_pairs(prices, min_corr=0.60, max_pvalue=0.10, max_half_life=90.0)

    print("\nTop Cointegrated Pairs Ranked by Statistical Significance:")
    print(coint_df.to_string(index=False))

    # 3. Johansen Cointegration Test on Top Energy Basket
    basket = ["CVX", "XOM", "COP"]
    if all(col in prices.columns for col in basket):
        print_banner("Johansen Multivariate Cointegration Test (Energy Basket: CVX, XOM, COP)")
        johansen_res = test_johansen(prices[basket])
        print(f"Johansen Trace Rank (95% confidence): {johansen_res['trace_rank']} cointegrating relationship(s)")
        print(f"Johansen Eigen Rank (95% confidence): {johansen_res['eigen_rank']} cointegrating relationship(s)")

    # 4. Select Top Pair for Analysis
    best_pair = coint_df.iloc[0]
    ticker_y = best_pair["Asset_Y"]
    ticker_x = best_pair["Asset_X"]
    pair_label = f"{ticker_y}-{ticker_x}"

    print_banner(f"Step 2: Spread Construction ({pair_label})")
    print(f"Selected Pair: Asset Y = {ticker_y}, Asset X = {ticker_x}")
    print(f"Cointegration p-value: {best_pair['Coint_PValue']:.5f} | Half-Life: {best_pair['Half_Life_Days']:.1f} days | Hurst: {best_pair.get('Hurst', 'N/A')}")

    py = prices[ticker_y]
    px = prices[ticker_x]

    # Model comparisons: OLS, TLS, and Kalman Filter
    ols_beta, ols_alpha, ols_spread = calculate_ols_spread(py, px)
    tls_beta, tls_alpha, tls_spread = calculate_tls_spread(py, px)
    kalman_res = calculate_kalman_spread(py, px, delta=1e-4, R=1e-3, adaptive=True)
    kf_beta = kalman_res["beta"]
    kf_spread = kalman_res["spread"]

    print(f"Static OLS Beta : {ols_beta:.4f} (Intercept: {ols_alpha:.4f})")
    print(f"Total Least Sq : {tls_beta:.4f} (Intercept: {tls_alpha:.4f})")
    print(f"Kalman Filter  : [{kf_beta.min():.4f}, {kf_beta.max():.4f}], Final Beta: {kf_beta.iloc[-1]:.4f}")

    # Select hedge ratio for backtest
    if args.hedge == "ols":
        chosen_beta = pd.Series(ols_beta, index=py.index)
        chosen_spread = ols_spread
    elif args.hedge == "tls":
        chosen_beta = pd.Series(tls_beta, index=py.index)
        chosen_spread = tls_spread
    else:
        chosen_beta = kf_beta
        chosen_spread = kf_spread

    # Rolling normalised z-scores
    zscore_df = calculate_rolling_zscore(chosen_spread, window=30)

    # 5. Signal Generation
    print_banner(f"Step 3: Signal Generation ({pair_label})")
    entry_thresh = 2.0
    exit_thresh = 0.0
    stop_loss_thresh = 4.0

    signals_df = generate_trading_signals(
        zscore_df["zscore"],
        entry_threshold=entry_thresh,
        exit_threshold=exit_thresh,
        stop_loss_threshold=stop_loss_thresh
    )

    # 6. Backtest with Institutional Friction
    print_banner(f"Step 4: Backtesting with Transaction Costs ({pair_label})")
    initial_cap = 100000.0
    backtest_res = run_backtest(
        price_y=py,
        price_x=px,
        hedge_ratio=chosen_beta,
        positions=signals_df["position"],
        initial_capital=initial_cap,
        sizing_mode=args.sizing,
        commission_bps=5.0,
        slippage_bps=5.0,
        borrow_rate_bps=100.0
    )

    daily_df = backtest_res["daily_df"]
    trades_df = backtest_res["trades"]

    metrics = calculate_performance_metrics(
        daily_returns=daily_df["daily_return"],
        equity_series=daily_df["equity"],
        trades_df=trades_df,
        daily_df=daily_df
    )

    print_banner(f"Performance Metrics: {pair_label}")
    for k, v in metrics.items():
        print(f" {k:<32}: {v}")

    # 7. Generate Visualisations & Interactive HTML Dashboard
    print_banner("Step 5: Visualisation & Dashboard Generation")
    plot_normalised_prices(prices, ticker_y, ticker_x, save_path="plots/normalised_prices.png")
    plot_spread_and_hedge_ratio(
        dates=prices.index,
        spread_ols=ols_spread,
        spread_kalman=kf_spread,
        beta_kalman=kf_beta,
        beta_ols=ols_beta,
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        save_path="plots/spread_and_hedge_ratio.png"
    )
    plot_zscore_and_signals(
        zscore_df=zscore_df,
        signals_df=signals_df,
        entry_threshold=entry_thresh,
        exit_threshold=exit_thresh,
        stop_loss_threshold=stop_loss_thresh,
        pair_label=pair_label,
        save_path="plots/zscore_signals.png"
    )
    norm_bench = 0.5 * (py / py.iloc[0]) + 0.5 * (px / px.iloc[0])
    plot_equity_curve_and_drawdown(
        daily_df=daily_df,
        initial_capital=initial_cap,
        benchmark_equity=norm_bench * initial_cap,
        save_path="plots/equity_curve.png"
    )

    generate_html_dashboard(
        coint_df=coint_df,
        pair_y=ticker_y,
        pair_x=ticker_x,
        prices=prices,
        kf_beta=kf_beta,
        tls_beta=tls_beta,
        ols_beta=ols_beta,
        spread=chosen_spread,
        zscore_df=zscore_df,
        signals_df=signals_df,
        backtest_res=backtest_res,
        metrics=metrics,
        output_path="dashboard.html"
    )
    print(" Saved: dashboard.html (Interactive Plotly Dashboard)")

    # 8. Multi-Pair Portfolio & Walk-Forward Summary
    print_banner("Step 6: Multi-Pair Portfolio & Walk-Forward Summary")
    n_pairs = min(3, len(coint_df))
    portfolio_res = run_multipair_portfolio(prices, coint_df, max_pairs=n_pairs, initial_capital=initial_cap)
    port_m = portfolio_res["portfolio_metrics"]
    print(f"Multi-Pair Top {n_pairs} Portfolio -> Total Return: {port_m.get('Total Return (%)', 'N/A')}% | Sharpe: {port_m.get('Sharpe Ratio', 'N/A')} | Max DD: {port_m.get('Max Drawdown (%)', 'N/A')}%")

    wf_res = run_walk_forward_validation(prices, in_sample_len=252, out_of_sample_len=63)
    wf_m = wf_res["stitched_metrics"]
    print(f"Walk-Forward Out-of-Sample -> Total Return: {wf_m.get('Total Return (%)', 'N/A')}% | CAGR: {wf_m.get('CAGR (%)', 'N/A')}% | Sharpe: {wf_m.get('Sharpe Ratio', 'N/A')}")

    print_banner("Execution Completed Successfully")


if __name__ == "__main__":
    main()
