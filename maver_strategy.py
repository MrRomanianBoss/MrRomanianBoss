"""
MAVER — Multi-Asset Volatility Expansion Reversal
Production-grade vectorized backtester for Futures Prop-Firm accounts.

Supported instruments : ES, NQ, CL, GC, YM, RTY
Timeframe             : 5-minute OHLCV
Regime filter         : ADX(14) < 25  →  mean-reversion only
Entry signals         : Close outside 2.5-σ Bollinger Band (20-period)
Exit logic            : 1.5 × ATR(14) SL / TP  →  reversion confirmed at 20-SMA
Position sizing       : 0.35 % max-risk on $50 000 account  ($175 / trade)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  Contract specifications ($ value per 1 point)
# ──────────────────────────────────────────────
CONTRACT_SPECS: Dict[str, float] = {
    "ES":  50.0,   # E-mini S&P 500
    "NQ":  20.0,   # E-mini NASDAQ-100
    "CL":1_000.0,  # Crude Oil
    "GC":  100.0,  # Gold
    "YM":    5.0,  # E-mini Dow
    "RTY":  50.0,  # E-mini Russell 2000
}

# ──────────────────────────────────────────────
#  Strategy parameters (all immutable defaults)
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class StrategyParams:
    account_size:    float = 50_000.0
    max_risk_pct:    float = 0.0035       # 0.35 %
    bb_period:       int   = 20
    bb_std:          float = 2.5
    adx_period:      int   = 14
    atr_period:      int   = 14
    atr_sl_mult:     float = 1.5
    atr_tp_mult:     float = 1.5          # 1:1 R:R minimum
    adx_threshold:   float = 25.0
    max_drawdown_pct:float = 0.05         # 5 % prop-firm hard limit
    commission_per_side: float = 2.25     # NinjaTrader / Rithmic typical


PARAMS = StrategyParams()


# ──────────────────────────────────────────────
#  Indicator layer  (purely vectorised, no loops)
# ──────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, params: StrategyParams = PARAMS) -> pd.DataFrame:
    """
    Adds all required indicators in-place and returns the enriched DataFrame.
    Input columns required: open, high, low, close, volume  (case-insensitive).
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bbands = ta.bbands(df["close"], length=params.bb_period, std=params.bb_std)
    df["bb_upper"] = bbands[f"BBU_{params.bb_period}_{params.bb_std}"]
    df["bb_lower"] = bbands[f"BBL_{params.bb_period}_{params.bb_std}"]
    df["bb_mid"]   = bbands[f"BBM_{params.bb_period}_{params.bb_std}"]   # = 20-SMA

    # ── ATR ─────────────────────────────────────────────────────────────────
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=params.atr_period)

    # ── ADX ─────────────────────────────────────────────────────────────────
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=params.adx_period)
    df["adx"] = adx_df[f"ADX_{params.adx_period}"]

    return df


# ──────────────────────────────────────────────
#  Signal generation  (vectorised boolean masks)
# ──────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, params: StrategyParams = PARAMS) -> pd.DataFrame:
    """
    Adds signal columns:
      signal   →  +1 long, -1 short, 0 no trade
      sl_price →  stop-loss price
      tp_price →  take-profit price
    """
    df = df.copy()

    regime_ok = df["adx"] < params.adx_threshold          # mean-reversion regime

    long_entry  = regime_ok & (df["close"] < df["bb_lower"])
    short_entry = regime_ok & (df["close"] > df["bb_upper"])

    df["signal"] = np.where(long_entry, 1, np.where(short_entry, -1, 0))

    # SL / TP anchored to ATR at signal bar
    atr_sl = df["atr"] * params.atr_sl_mult
    atr_tp = df["atr"] * params.atr_tp_mult

    df["sl_price"] = np.where(
        df["signal"] ==  1, df["close"] - atr_sl,
        np.where(df["signal"] == -1, df["close"] + atr_sl, np.nan)
    )
    df["tp_price"] = np.where(
        df["signal"] ==  1, df["close"] + atr_tp,
        np.where(df["signal"] == -1, df["close"] - atr_tp, np.nan)
    )

    return df


# ──────────────────────────────────────────────
#  Position sizing
# ──────────────────────────────────────────────

def calc_position_size(
    entry_price: float,
    sl_price: float,
    symbol: str,
    params: StrategyParams = PARAMS,
) -> int:
    """
    Returns the number of contracts to trade (minimum 1, rounded down).
    Risk per trade = account_size × max_risk_pct
    """
    max_dollar_risk = params.account_size * params.max_risk_pct   # $175
    point_risk      = abs(entry_price - sl_price)
    tick_value      = CONTRACT_SPECS.get(symbol, 50.0)

    if point_risk == 0:
        return 1

    contracts = max_dollar_risk / (point_risk * tick_value)
    return max(1, int(contracts))


# ──────────────────────────────────────────────
#  Vectorised back-tester
# ──────────────────────────────────────────────

@dataclass
class Trade:
    symbol:     str
    entry_time: pd.Timestamp
    exit_time:  pd.Timestamp
    direction:  int        # +1 long / -1 short
    entry:      float
    exit_price: float
    sl:         float
    tp:         float
    contracts:  int
    pnl:        float
    duration_bars: int


def _simulate_symbol(
    df: pd.DataFrame,
    symbol: str,
    params: StrategyParams = PARAMS,
) -> List[Trade]:
    """
    Bar-by-bar trade simulation on a single symbol's enriched DataFrame.
    Avoids scalping by exiting only at SL, TP, or reversion to the 20-SMA.
    One open trade at a time per symbol.
    """
    trades: List[Trade] = []
    in_trade   = False
    entry_bar  = None
    direction  = 0
    entry_px   = sl_px = tp_px = 0.0
    num_lots   = 1

    tick_val = CONTRACT_SPECS.get(symbol, 50.0)
    commission = params.commission_per_side * 2   # round-trip

    df = df.reset_index(drop=False)   # keep original timestamp in 'index' col if DatetimeIndex

    # Normalise timestamp column name
    ts_col = "index" if "index" in df.columns else df.columns[0]

    for i in range(1, len(df)):
        row     = df.iloc[i]
        prev    = df.iloc[i - 1]

        if in_trade:
            hi, lo, cl = row["high"], row["low"], row["close"]
            mid        = row["bb_mid"]

            hit_sl = (direction ==  1 and lo <= sl_px) or \
                     (direction == -1 and hi >= sl_px)
            hit_tp = (direction ==  1 and hi >= tp_px) or \
                     (direction == -1 and lo <= tp_px)
            hit_mean = (direction ==  1 and cl >= mid) or \
                       (direction == -1 and cl <= mid)

            if hit_sl or hit_tp or hit_mean:
                exit_px = (sl_px if hit_sl else
                           tp_px if hit_tp else
                           mid)

                raw_pnl = (exit_px - entry_px) * direction * num_lots * tick_val
                net_pnl = raw_pnl - commission * num_lots

                trades.append(Trade(
                    symbol     = symbol,
                    entry_time = df.iloc[entry_bar][ts_col],
                    exit_time  = row[ts_col],
                    direction  = direction,
                    entry      = entry_px,
                    exit_price = exit_px,
                    sl         = sl_px,
                    tp         = tp_px,
                    contracts  = num_lots,
                    pnl        = net_pnl,
                    duration_bars = i - entry_bar,
                ))
                in_trade = False

        elif prev["signal"] != 0:
            sig       = int(prev["signal"])
            entry_px  = row["open"]             # fill at next-bar open (realistic)
            sl_px     = prev["sl_price"]
            tp_px     = prev["tp_price"]
            num_lots  = calc_position_size(entry_px, sl_px, symbol, params)
            direction = sig
            entry_bar = i
            in_trade  = True

    return trades


def run_backtest(
    data: Dict[str, pd.DataFrame],
    params: StrategyParams = PARAMS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full MAVER backtest across all symbols.

    Parameters
    ----------
    data : dict  {symbol: raw_OHLCV_DataFrame}

    Returns
    -------
    trades_df  : per-trade log
    equity_df  : daily equity curve (all symbols combined)
    """
    all_trades: List[Trade] = []

    for symbol, raw_df in data.items():
        enriched = compute_indicators(raw_df, params)
        signaled = generate_signals(enriched, params)
        trades   = _simulate_symbol(signaled, symbol, params)
        all_trades.extend(trades)

    trades_df = pd.DataFrame([t.__dict__ for t in all_trades])

    if trades_df.empty:
        print("No trades generated — check data quality or parameter ranges.")
        return trades_df, pd.DataFrame()

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    # ── Equity curve ────────────────────────────────────────────────────────
    trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
    trades_df["equity"]         = params.account_size + trades_df["cumulative_pnl"]

    equity_df = (
        trades_df.set_index("exit_time")["equity"]
        .resample("1D").last()
        .ffill()
    )

    return trades_df, equity_df


# ──────────────────────────────────────────────
#  Drawdown analysis
# ──────────────────────────────────────────────

def drawdown_analysis(equity_series: pd.Series) -> Dict[str, float]:
    """
    Computes peak-to-trough drawdown statistics.

    Returns
    -------
    dict with keys: max_drawdown_pct, max_drawdown_dollar,
                    avg_drawdown_pct, longest_drawdown_bars
    """
    roll_max   = equity_series.cummax()
    drawdown   = (equity_series - roll_max) / roll_max

    max_dd_pct    = drawdown.min()
    max_dd_dollar = (equity_series - roll_max).min()

    # Longest consecutive drawdown period (bars in drawdown)
    in_dd      = (drawdown < 0).astype(int)
    dd_lengths = []
    current    = 0
    for v in in_dd:
        if v:
            current += 1
        else:
            if current:
                dd_lengths.append(current)
            current = 0
    if current:
        dd_lengths.append(current)

    return {
        "max_drawdown_pct":     round(max_dd_pct * 100, 2),
        "max_drawdown_dollar":  round(max_dd_dollar, 2),
        "avg_drawdown_pct":     round(drawdown[drawdown < 0].mean() * 100, 2) if (drawdown < 0).any() else 0.0,
        "longest_drawdown_bars": max(dd_lengths) if dd_lengths else 0,
    }


# ──────────────────────────────────────────────
#  Performance summary
# ──────────────────────────────────────────────

def performance_summary(
    trades_df: pd.DataFrame,
    equity_df: pd.Series,
    params: StrategyParams = PARAMS,
) -> Dict[str, object]:
    """
    Produces a comprehensive performance report dictionary.
    """
    if trades_df.empty:
        return {}

    wins  = trades_df[trades_df["pnl"] > 0]
    loses = trades_df[trades_df["pnl"] <= 0]

    win_rate      = len(wins) / len(trades_df)
    avg_win       = wins["pnl"].mean()   if not wins.empty  else 0.0
    avg_loss      = loses["pnl"].mean()  if not loses.empty else 0.0
    profit_factor = (wins["pnl"].sum() / abs(loses["pnl"].sum())
                     if not loses.empty and loses["pnl"].sum() != 0 else np.inf)

    avg_dur_bars  = trades_df["duration_bars"].mean()

    dd_stats = drawdown_analysis(equity_df) if not equity_df.empty else {}

    net_pnl    = trades_df["pnl"].sum()
    total_days = max((equity_df.index[-1] - equity_df.index[0]).days, 1) if not equity_df.empty else 1
    cagr       = ((params.account_size + net_pnl) / params.account_size) ** (365 / total_days) - 1

    sharpe_ratio = (
        (trades_df["pnl"].mean() / trades_df["pnl"].std()) * np.sqrt(252)
        if trades_df["pnl"].std() > 0 else 0.0
    )

    prop_firm_pass = (
        dd_stats.get("max_drawdown_pct", -100) > -params.max_drawdown_pct * 100
    )

    return {
        "total_trades":         len(trades_df),
        "win_rate_pct":         round(win_rate * 100, 1),
        "avg_win_dollar":       round(avg_win, 2),
        "avg_loss_dollar":      round(avg_loss, 2),
        "profit_factor":        round(profit_factor, 2),
        "net_pnl_dollar":       round(net_pnl, 2),
        "avg_trade_duration_bars": round(avg_dur_bars, 1),
        "annualised_return_pct":round(cagr * 100, 2),
        "sharpe_ratio":         round(sharpe_ratio, 2),
        "prop_firm_pass":       prop_firm_pass,
        **dd_stats,
    }


# ──────────────────────────────────────────────
#  Pretty-print helper
# ──────────────────────────────────────────────

def print_report(summary: Dict[str, object]) -> None:
    width = 46
    print("=" * width)
    print("  MAVER Strategy — Performance Report")
    print("=" * width)
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        if isinstance(v, bool):
            val = "YES ✓" if v else "NO  ✗"
        elif isinstance(v, float):
            val = f"{v:,.2f}"
        else:
            val = str(v)
        print(f"  {label:<32} {val:>10}")
    print("=" * width)


if __name__ == "__main__":
    # ── Quick smoke-test with synthetic data ────────────────────────────────
    import warnings; warnings.filterwarnings("ignore")

    rng = np.random.default_rng(42)

    def _make_synthetic(n: int = 5_000, base: float = 4_500.0) -> pd.DataFrame:
        """Geometric Brownian Motion + mean-reverting noise to exercise both regimes."""
        log_ret = rng.normal(0, 0.0008, n)
        close   = base * np.exp(np.cumsum(log_ret))
        hi      = close * (1 + rng.uniform(0.0002, 0.0015, n))
        lo      = close * (1 - rng.uniform(0.0002, 0.0015, n))
        op      = close * (1 + rng.normal(0, 0.0003, n))
        vol     = rng.integers(100, 3000, n)
        idx     = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
        return pd.DataFrame({"open": op, "high": hi, "low": lo,
                             "close": close, "volume": vol}, index=idx)

    synthetic_data = {sym: _make_synthetic() for sym in CONTRACT_SPECS}

    trades_df, equity_df = run_backtest(synthetic_data)
    summary = performance_summary(trades_df, equity_df)
    print_report(summary)

    if not trades_df.empty:
        print(f"\nSample trades (first 5):\n{trades_df[['symbol','direction','entry','exit_price','pnl','duration_bars']].head()}")
