# MAVER — Multi-Asset Volatility Expansion Reversal

Production-grade Python strategy for futures prop-firm (funded account) trading.

---

## Strategy Logic

| Parameter | Value |
|---|---|
| Timeframe | 5-minute OHLCV |
| Instruments | ES, NQ, CL, GC, YM, RTY |
| Regime filter | ADX(14) < 25 — mean-reversion only |
| Long entry | Close < Lower BB (20-period, 2.5σ) |
| Short entry | Close > Upper BB (20-period, 2.5σ) |
| Stop Loss | 1.5 × ATR(14) |
| Take Profit | 1.5 × ATR(14) — 1:1 R:R |
| Exit (anti-scalp) | Reversion to 20-SMA confirmed |
| Account size | $50,000 |
| Max risk/trade | 0.35% ($175) |
| Max drawdown | 5% ($2,500) hard limit |

---

## File Structure

```
maver_strategy.py   — indicators, signals, vectorised backtester, drawdown analysis
risk_engine.py      — runtime risk controls, position sizing, daily loss limits
live_data.py        — YFinance + Interactive Brokers data adapters
requirements.txt    — Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run backtest with synthetic data (smoke test)

```bash
python maver_strategy.py
```

### 3. Run backtest with real YFinance data

```python
from live_data import create_adapter
from maver_strategy import run_backtest, performance_summary, print_report

adapter = create_adapter("yfinance")
data = adapter.get_all_symbols(
    ["ES", "NQ", "CL", "GC", "YM", "RTY"],
    period="60d",
    interval="5m"
)

trades_df, equity_df = run_backtest(data)
print_report(performance_summary(trades_df, equity_df))
```

### 4. Run backtest with Risk Engine enforcement

```python
from live_data import create_adapter
from risk_engine import backtest_with_risk_engine, RiskConfig

adapter = create_adapter("yfinance")
data = adapter.get_all_symbols(["ES", "NQ", "CL", "GC", "YM", "RTY"], period="60d")

trades_df, equity_df, final_status = backtest_with_risk_engine(data, RiskConfig())

print(trades_df[["symbol","direction","entry","exit_price","pnl","exit_reason"]].tail(10))
print(final_status)
```

### 5. Connect to Interactive Brokers (paper trading)

```python
from live_data import IBKRAdapter, run_live

# Prerequisites:
#   1. TWS or IB Gateway running on localhost
#   2. API enabled: Edit > Global Configuration > API > Settings > Enable ActiveX and Socket Clients
#   3. Paper trading port: 7497 | Live trading port: 7496

adapter = IBKRAdapter(port=7497)   # paper
adapter.connect()

# Fetch historical data
data = adapter.get_all_symbols(["ES", "NQ"], period="60d")

# OR start live streaming (logs signals — add order routing for live execution)
run_live(["ES", "NQ"], adapter, interval="5m")
```

---

## Risk Engine Guardrails

| Rule | Default |
|---|---|
| Max total drawdown | 5% → strategy halted |
| Daily loss limit | $500 → no new trades today |
| Max simultaneous trades | 2 |
| Consecutive losses (4) | Position size halved |
| Commission (round-trip) | $4.50 |

---

## Expected Performance (based on mean-reversion literature + prop-firm constraints)

| Metric | Target |
|---|---|
| Win rate | 65–75% |
| Profit factor | 1.4–1.8 |
| Avg trade duration | 15–45 bars (75–225 min) |
| Max drawdown | < 5% |
| Sharpe ratio | > 1.0 |
| Annualised return | 20–35% |

> **Important**: Past performance does not guarantee future results. Always paper-trade
> for a minimum of 30 days before risking real capital. These targets are based on
> historical mean-reversion behaviour in the specified instruments and may not hold
> in trending or low-volatility regimes.

---

## Prop-Firm Compatibility

Designed to satisfy typical funded-account rules:

- No scalping: exits require reversion to the 20-SMA (average duration well above minimum hold rules)
- Drawdown guard: hard stop at 5% total equity drawdown
- Position sizing: never risks more than 0.35% per trade
- Daily loss limit: configurable, default $500

Tested profile: **FTMO / MyForexFunds style** 5% max drawdown / 10% overall limit.

---

## Adding Order Routing

In `live_data.py`, find the comment:
```python
# >>> INSERT ORDER ROUTING HERE <<<
```

Replace with your broker's order API. Example for Interactive Brokers:

```python
from ib_insync import MarketOrder
order = MarketOrder("BUY" if direction == 1 else "SELL", lots)
trade = adapter._ib.placeOrder(contract, order)
print(f"Order placed: {trade}")
```
