"""
Interactive dashboard generator for statistical arbitrage pairs trading.
Produces a self-contained HTML dashboard with interactive Plotly.js charts.
"""

from __future__ import annotations

import json
from typing import Dict, Any
import pandas as pd


def generate_html_dashboard(
    coint_df: pd.DataFrame,
    pair_y: str,
    pair_x: str,
    prices: pd.DataFrame,
    kf_beta: pd.Series,
    tls_beta: float,
    ols_beta: float,
    spread: pd.Series,
    zscore_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    backtest_res: Dict[str, Any],
    metrics: Dict[str, Any],
    output_path: str = "dashboard.html"
) -> None:
    """Generate a self-contained interactive HTML dashboard."""
    dates = [d.strftime("%Y-%m-%d") for d in prices.index]
    daily = backtest_res["daily_df"]

    # Prepare chart data series
    py_norm = ((prices[pair_y] / prices[pair_y].iloc[0]) * 100.0).round(2).tolist()
    px_norm = ((prices[pair_x] / prices[pair_x].iloc[0]) * 100.0).round(2).tolist()

    equity = daily["equity"].round(2).tolist()
    peak = daily["equity"].cummax()
    drawdown = (((daily["equity"] - peak) / peak) * 100.0).round(2).tolist()

    z_vals = zscore_df["zscore"].round(2).tolist()
    beta_vals = kf_beta.round(4).tolist()
    spread_vals = spread.round(3).tolist()

    sig = signals_df["signal"].values
    long_dates = [dates[i] for i in range(len(dates)) if sig[i] == 1]
    long_z = [z_vals[i] for i in range(len(dates)) if sig[i] == 1]
    short_dates = [dates[i] for i in range(len(dates)) if sig[i] == -1]
    short_z = [z_vals[i] for i in range(len(dates)) if sig[i] == -1]

    # Metrics cards HTML
    metrics_cards = "".join([
        f"""<div class="card"><div class="card-title">{k}</div><div class="card-value">{v}</div></div>"""
        for k, v in list(metrics.items())[:8]
    ])

    # Table HTML
    table_rows = "".join([
        f"<tr><td>{r['Asset_Y']}</td><td>{r['Asset_X']}</td><td>{r['Correlation']}</td><td>{r['Coint_PValue']}</td><td>{r.get('Hurst', 'N/A')}</td><td>{r['Half_Life_Days']}d</td><td>{r['Hedge_Ratio']}</td></tr>"
        for _, r in coint_df.head(5).iterrows()
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Statistical Arbitrage Dashboard ({pair_y}-{pair_x})</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f141c; color: #e1e7ed; margin: 0; padding: 24px; }}
  h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #8b98a5; font-size: 14px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1a222d; border: 1px solid #2d3846; border-radius: 8px; padding: 16px; }}
  .card-title {{ font-size: 12px; color: #8b98a5; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card-value {{ font-size: 22px; font-weight: 700; margin-top: 6px; color: #4ade80; }}
  .chart-box {{ background: #1a222d; border: 1px solid #2d3846; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #2d3846; }}
  th {{ color: #8b98a5; font-weight: 600; }}
</style>
</head>
<body>
<h1>Institutional Statistical Arbitrage Pairs Trading</h1>
<div class="subtitle">Selected Pair: {pair_y} / {pair_x} | OLS Beta: {ols_beta:.3f} | TLS Beta: {tls_beta:.3f} | Dynamic Kalman Filter</div>

<div class="grid">{metrics_cards}</div>

<div class="chart-box">
  <div id="chart-prices" style="height: 360px;"></div>
</div>

<div class="chart-box">
  <div id="chart-spread-beta" style="height: 400px;"></div>
</div>

<div class="chart-box">
  <div id="chart-zscore" style="height: 360px;"></div>
</div>

<div class="chart-box">
  <div id="chart-equity" style="height: 400px;"></div>
</div>

<div class="chart-box">
  <h3 style="margin-top:0; font-size:16px;">Top Cointegrated Universe Pairs</h3>
  <table>
    <thead><tr><th>Asset Y</th><th>Asset X</th><th>Correlation</th><th>Coint p-value</th><th>Hurst</th><th>Half-Life</th><th>Beta</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<script>
const dates = {json.dumps(dates)};
const darkLayout = {{
  paper_bgcolor: '#1a222d', plot_bgcolor: '#1a222d',
  font: {{ color: '#8b98a5' }}, margin: {{ t: 30, b: 40, l: 50, r: 20 }},
  xaxis: {{ gridcolor: '#263342' }}, yaxis: {{ gridcolor: '#263342' }}
}};

Plotly.newPlot('chart-prices', [
  {{ x: dates, y: {json.dumps(py_norm)}, name: '{pair_y} (Norm)', line: {{ color: '#38bdf8' }} }},
  {{ x: dates, y: {json.dumps(px_norm)}, name: '{pair_x} (Norm)', line: {{ color: '#f59e0b' }} }}
], {{ ...darkLayout, title: 'Normalised Prices ({pair_y} vs {pair_x})' }});

Plotly.newPlot('chart-spread-beta', [
  {{ x: dates, y: {json.dumps(spread_vals)}, name: 'Kalman Spread', line: {{ color: '#34d399' }} }},
  {{ x: dates, y: {json.dumps(beta_vals)}, name: 'Kalman Beta', yaxis: 'y2', line: {{ color: '#f43f5e', dash: 'dot' }} }}
], {{
  ...darkLayout, title: 'Dynamic Spread & Adaptive Kalman Beta',
  yaxis2: {{ title: 'Beta', overlaying: 'y', side: 'right', gridcolor: '#263342' }}
}});

Plotly.newPlot('chart-zscore', [
  {{ x: dates, y: {json.dumps(z_vals)}, name: 'Z-Score', line: {{ color: '#818cf8' }} }},
  {{ x: {json.dumps(long_dates)}, y: {json.dumps(long_z)}, mode: 'markers', name: 'Long Entry (-2σ)', marker: {{ color: '#22c55e', symbol: 'triangle-up', size: 10 }} }},
  {{ x: {json.dumps(short_dates)}, y: {json.dumps(short_z)}, mode: 'markers', name: 'Short Entry (+2σ)', marker: {{ color: '#ef4444', symbol: 'triangle-down', size: 10 }} }}
], {{
  ...darkLayout, title: 'Z-Score & Trading Execution Signals',
  shapes: [
    {{ type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 2, y1: 2, line: {{ color: '#ef4444', dash: 'dash' }} }},
    {{ type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: -2, y1: -2, line: {{ color: '#22c55e', dash: 'dash' }} }},
    {{ type: 'line', x0: dates[0], x1: dates[dates.length-1], y0: 0, y1: 0, line: {{ color: '#64748b', dash: 'dot' }} }}
  ]
}});

Plotly.newPlot('chart-equity', [
  {{ x: dates, y: {json.dumps(equity)}, name: 'Strategy Equity ($)', line: {{ color: '#4ade80' }} }},
  {{ x: dates, y: {json.dumps(drawdown)}, name: 'Drawdown (%)', yaxis: 'y2', fill: 'tozeroy', line: {{ color: '#f87171' }} }}
], {{
  ...darkLayout, title: 'Portfolio Equity & Underwater Drawdown',
  yaxis: {{ title: 'Equity ($)', gridcolor: '#263342' }},
  yaxis2: {{ title: 'Drawdown (%)', overlaying: 'y', side: 'right', gridcolor: '#263342' }}
}});
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
