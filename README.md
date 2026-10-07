# Statistical Arbitrage Pairs Trading

An institutional-grade quantitative framework for statistical arbitrage pairs trading in Python. The system screens candidate equities for stationarity and cointegration, constructs static OLS and dynamic Kalman Filter state-space hedge ratios, generates mean-reverting z-score trading signals, and performs rigorous backtests accounting for broker commissions, bid-ask slippage, and short equity borrow fees.

---

## Key Features

1. **Robust Universe Screening**:
   - Pearson correlation thresholding ($r \ge 0.70$).
   - Engle-Granger two-step cointegration testing ($p < 0.05$) via `statsmodels`.
   - Augmented Dickey-Fuller (ADF) residual stationarity verification.
   - Mean-reversion half-life estimation via Ornstein-Uhlenbeck process modelling ($5 \le T_{1/2} \le 60$ days).

2. **Spread & Hedge Ratio Estimation**:
   - **Static OLS**: Benchmark linear regression $Y = \alpha + \beta X + \epsilon$.
   - **Dynamic Kalman Filter**: Real-time state-space parameter updating ($\theta_t = [\alpha_t, \beta_t]^T$) adapting to structural market regime shifts without lookahead bias.
   - **Normalised Rolling Z-Scores**: Moving window standardisation ($Z_t = \frac{S_t - \mu_t}{\sigma_t}$) for stationary spread tracking.

3. **Execution & Risk Management**:
   - Entry thresholds: Long spread at $Z \le -2.0\sigma$, Short spread at $Z \ge +2.0\sigma$.
   - Mean-reversion exit: Position liquidation upon crossing $Z = 0.0\sigma$.
   - Structural breakdown stop-loss: Liquidation at $|Z| \ge 4.0\sigma$ to protect capital against regime divergence.
   - **Strict 1-Day Execution Lag**: Signal calculated at day $t$ close is transacted at day $t+1$, eliminating lookahead bias.

4. **Realistic Market Microstructure & Costs**:
   - Brokerage commission (5 bps).
   - Market execution slippage (5 bps).
   - Annualised borrow fee on short equity market value (100 bps).
   - Cash mark-to-market, capital utilisation, and gross exposure tracking.

5. **Advanced Capabilities**:
   - **Multi-Pair Portfolio Allocation**: Combines uncorrelated cointegrated pairs into a single market-neutral portfolio.
   - **Walk-Forward Validation**: Rolling 1-year in-sample calibration and 3-month out-of-sample execution to verify real-world robustness.

---

## Directory Structure

```text
Statistical Arbitrage/
├── data/
│   ├── prices.csv              # Historical prices for candidate universe
│   └── data_loader.py          # Data ingestion with caching & synthetic fallback
├── src/
│   ├── __init__.py
│   ├── pair_selection.py       # Correlation, Engle-Granger cointegration, ADF, half-life
│   ├── spread.py               # Static OLS, dynamic Kalman Filter, rolling z-score
│   ├── signals.py              # Z-score entry, exit, and stop-loss signal generation
│   ├── backtest.py             # Event-driven portfolio accounting & friction modelling
│   ├── metrics.py              # Sharpe, Sortino, Calmar, max drawdown, win rate
│   ├── portfolio.py            # Multi-pair allocation & walk-forward validation
│   └── visualisation.py        # High-resolution publication charts
├── notebooks/
│   └── analysis.ipynb          # Interactive quantitative walkthrough
├── plots/                      # Rendered strategy charts
│   ├── normalised_prices.png
│   ├── spread_and_hedge_ratio.png
│   ├── zscore_signals.png
│   ├── equity_curve.png
│   └── multipair_equity_curve.png
├── tests/                      # Automated unit test suite
│   ├── test_cointegration.py
│   ├── test_spread.py
│   ├── test_signals.py
│   ├── test_backtest.py
│   └── test_metrics.py
├── scripts/
│   └── create_notebook.py
├── requirements.txt
├── README.md
└── main.py                     # Primary pipeline execution script
```

---

## Mathematical Formulation

### 1. Engle-Granger Cointegration
Two non-stationary $I(1)$ series $Y_t$ and $X_t$ are cointegrated of order $(1, 1)$ if a linear combination is stationary $I(0)$:
$$Y_t = \alpha + \beta X_t + \epsilon_t, \quad \epsilon_t \sim I(0)$$

### 2. Ornstein-Uhlenbeck Process & Mean Reversion Half-Life
The continuous-time mean-reverting spread is modelled as:
$$d S_t = \theta (\mu - S_t) dt + \sigma dW_t$$
Discretised via regression:
$$\Delta S_t = \gamma S_{t-1} + c + \eta_t$$
where $\gamma = -(1 - e^{-\theta \Delta t})$. The half-life is:
$$T_{1/2} = -\frac{\ln(2)}{\gamma}$$

### 3. Dynamic Kalman Filter State-Space Formulation
State transition:
$$\begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix} = \begin{bmatrix} \alpha_{t-1} \\ \beta_{t-1} \end{bmatrix} + w_t, \quad w_t \sim \mathcal{N}\left(0, \frac{\delta}{1 - \delta} I_2\right)$$
Observation equation:
$$Y_t = \begin{bmatrix} 1 & X_t \end{bmatrix} \begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix} + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Strategy Pipeline
```bash
python main.py
```
This executes the entire pipeline:
- Screens candidate universe for cointegrated pairs.
- Estimates OLS and Kalman Filter hedge ratios.
- Generates z-score entry/exit/stop-loss signals.
- Runs the backtest with transaction costs.
- Calculates institutional performance metrics (Sharpe, CAGR, Drawdown, Win Rate).
- Generates publication-ready charts in `plots/`.
- Runs multi-pair portfolio aggregation and walk-forward validation.

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

### 4. Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
