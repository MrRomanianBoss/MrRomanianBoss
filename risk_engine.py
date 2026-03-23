"""
MAVER Risk Engine
=================
Enforces prop-firm drawdown rules, daily loss limits, and position-level
risk controls at runtime — both during backtesting and live trading.

Key guardrails
--------------
* Hard stop  : total equity drawdown ≥ 5 %  →  halt all new entries
* Daily loss  : single-day P&L ≤ −$500      →  halt entries for the day
* Consecutive losses : ≥ 4 in a row          →  reduce size to 50 %
* Exposure cap: no more than 2 open trades simultaneously across all symbols
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Risk parameters
# ──────────────────────────────────────────────

@dataclass
class RiskConfig:
    account_size:          float = 50_000.0
    max_drawdown_pct:      float = 0.05       # 5 % prop-firm hard limit
    daily_loss_limit:      float = 500.0      # $ per day
    max_open_trades:       int   = 2          # simultaneous positions
    consecutive_loss_cap:  int   = 4          # triggers half-size
    size_reduction_factor: float = 0.50       # applied after cap hit
    max_risk_per_trade_pct:float = 0.0035     # 0.35 %
    commission_rt:         float = 4.50       # round-trip $


# ──────────────────────────────────────────────
#  Runtime state tracker
# ──────────────────────────────────────────────

@dataclass
class RiskState:
    peak_equity:        float = 50_000.0
    current_equity:     float = 50_000.0
    daily_pnl:          float = 0.0
    daily_reset_date:   Optional[date] = None
    open_trades:        int   = 0
    consecutive_losses: int   = 0
    total_trades:       int   = 0
    total_wins:         int   = 0
    halted:             bool  = False          # master kill-switch
    halt_reason:        str   = ""


class RiskEngine:
    """
    Stateful risk engine that wraps strategy signals.

    Usage
    -----
    engine = RiskEngine(RiskConfig())

    # Before taking a trade:
    allowed, reason, lots = engine.check_entry(symbol, entry, sl, direction)

    # After trade closes:
    engine.record_trade(pnl)
    """

    def __init__(self, config: RiskConfig = RiskConfig()):
        self.cfg   = config
        self.state = RiskState(
            peak_equity    = config.account_size,
            current_equity = config.account_size,
        )
        self._trade_log: List[Dict] = []

    # ── Internal helpers ────────────────────────────────────────────────────

    def _reset_daily_if_needed(self, now: Optional[datetime] = None) -> None:
        today = (now or datetime.utcnow()).date()
        if self.state.daily_reset_date != today:
            self.state.daily_pnl        = 0.0
            self.state.daily_reset_date = today

    def _current_drawdown_pct(self) -> float:
        if self.state.peak_equity == 0:
            return 0.0
        return (self.state.current_equity - self.state.peak_equity) / self.state.peak_equity

    def _effective_risk_pct(self) -> float:
        """Halves risk after consecutive-loss cap is triggered."""
        base = self.cfg.max_risk_per_trade_pct
        if self.state.consecutive_losses >= self.cfg.consecutive_loss_cap:
            return base * self.cfg.size_reduction_factor
        return base

    # ── Public API ──────────────────────────────────────────────────────────

    def check_entry(
        self,
        symbol:      str,
        entry_price: float,
        sl_price:    float,
        direction:   int,              # +1 long / -1 short
        tick_value:  float = 50.0,
        now:         Optional[datetime] = None,
    ) -> Tuple[bool, str, int]:
        """
        Evaluate whether a new trade entry is permitted.

        Returns
        -------
        (allowed: bool, reason: str, contracts: int)
        """
        self._reset_daily_if_needed(now)

        # ── Master halt ─────────────────────────────────────────────────────
        if self.state.halted:
            return False, f"HALTED: {self.state.halt_reason}", 0

        # ── Drawdown guard ──────────────────────────────────────────────────
        dd = self._current_drawdown_pct()
        if dd <= -self.cfg.max_drawdown_pct:
            self._halt(f"Max drawdown breached: {dd*100:.2f}%")
            return False, self.state.halt_reason, 0

        # ── Daily loss limit ────────────────────────────────────────────────
        if self.state.daily_pnl <= -self.cfg.daily_loss_limit:
            return False, f"Daily loss limit hit (${self.state.daily_pnl:.2f})", 0

        # ── Open trade cap ──────────────────────────────────────────────────
        if self.state.open_trades >= self.cfg.max_open_trades:
            return False, f"Max open trades ({self.cfg.max_open_trades}) reached", 0

        # ── Position size ───────────────────────────────────────────────────
        dollar_risk  = self.state.current_equity * self._effective_risk_pct()
        point_risk   = abs(entry_price - sl_price)
        contracts    = max(1, int(dollar_risk / (point_risk * tick_value))) if point_risk > 0 else 1

        self.state.open_trades += 1
        return True, "OK", contracts

    def record_trade(
        self,
        pnl:    float,
        symbol: str = "",
        now:    Optional[datetime] = None,
    ) -> None:
        """
        Update engine state after a trade closes.
        Must be called for every closed trade.
        """
        self._reset_daily_if_needed(now)

        net_pnl = pnl - self.cfg.commission_rt
        self.state.current_equity += net_pnl
        self.state.daily_pnl      += net_pnl
        self.state.open_trades     = max(0, self.state.open_trades - 1)
        self.state.total_trades   += 1

        if net_pnl > 0:
            self.state.total_wins      += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        # Update peak equity
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity

        self._trade_log.append({
            "timestamp": now or datetime.utcnow(),
            "symbol":    symbol,
            "pnl":       net_pnl,
            "equity":    self.state.current_equity,
        })

        logger.debug(
            "Trade closed | PnL: $%.2f | Equity: $%.2f | DD: %.2f%%",
            net_pnl, self.state.current_equity,
            self._current_drawdown_pct() * 100,
        )

    def _halt(self, reason: str) -> None:
        self.state.halted      = True
        self.state.halt_reason = reason
        logger.critical("RISK ENGINE HALT: %s", reason)

    # ── Reporting ───────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "current_equity":       round(self.state.current_equity, 2),
            "peak_equity":          round(self.state.peak_equity, 2),
            "drawdown_pct":         round(self._current_drawdown_pct() * 100, 2),
            "daily_pnl":            round(self.state.daily_pnl, 2),
            "open_trades":          self.state.open_trades,
            "consecutive_losses":   self.state.consecutive_losses,
            "total_trades":         self.state.total_trades,
            "win_rate_pct":         round(
                self.state.total_wins / max(1, self.state.total_trades) * 100, 1
            ),
            "halted":               self.state.halted,
            "halt_reason":          self.state.halt_reason,
            "effective_risk_pct":   round(self._effective_risk_pct() * 100, 3),
        }

    def trade_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)

    def equity_curve(self) -> pd.Series:
        df = self.trade_log_df()
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index("timestamp")["equity"]


# ──────────────────────────────────────────────
#  Integrated back-test wrapper  (uses both
#  maver_strategy.py and RiskEngine together)
# ──────────────────────────────────────────────

def backtest_with_risk_engine(
    data:   Dict[str, pd.DataFrame],
    config: RiskConfig = RiskConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Drop-in replacement for maver_strategy.run_backtest that enforces
    all runtime risk limits during the simulation.

    Returns
    -------
    trades_df, equity_df, final_status
    """
    # Import here to avoid circular import at module level
    from maver_strategy import (
        compute_indicators,
        generate_signals,
        CONTRACT_SPECS,
        StrategyParams,
        performance_summary,
    )

    params = StrategyParams(
        account_size   = config.account_size,
        max_risk_pct   = config.max_risk_per_trade_pct,
    )

    engine = RiskEngine(config)
    records: List[Dict] = []

    # Prepare all symbol DataFrames
    enriched: Dict[str, pd.DataFrame] = {}
    for sym, raw_df in data.items():
        df = compute_indicators(raw_df, params)
        df = generate_signals(df, params)
        df = df.reset_index(drop=False)
        enriched[sym] = df

    # Build a unified event timeline
    events: List[Tuple[pd.Timestamp, str, int]] = []   # (time, symbol, bar_idx)
    for sym, df in enriched.items():
        ts_col = df.columns[0]
        for i, ts in enumerate(df[ts_col]):
            events.append((ts, sym, i))
    events.sort(key=lambda x: x[0])

    open_positions: Dict[str, Dict] = {}   # symbol → position info

    for ts, sym, i in events:
        df  = enriched[sym]
        row = df.iloc[i]

        # ── Check if any open position closes on this bar ────────────────
        if sym in open_positions:
            pos = open_positions[sym]
            hi, lo, cl, mid = row["high"], row["low"], row["close"], row["bb_mid"]
            d   = pos["direction"]

            hit_sl   = (d ==  1 and lo  <= pos["sl"]) or (d == -1 and hi >= pos["sl"])
            hit_tp   = (d ==  1 and hi  >= pos["tp"]) or (d == -1 and lo <= pos["tp"])
            hit_mean = (d ==  1 and cl  >= mid)        or (d == -1 and cl <= mid)

            if hit_sl or hit_tp or hit_mean:
                exit_px = pos["sl"] if hit_sl else pos["tp"] if hit_tp else mid
                raw_pnl = (exit_px - pos["entry"]) * d * pos["lots"] * pos["tick_val"]
                engine.record_trade(raw_pnl, sym, ts.to_pydatetime())

                records.append({
                    "symbol":       sym,
                    "entry_time":   pos["entry_time"],
                    "exit_time":    ts,
                    "direction":    d,
                    "entry":        pos["entry"],
                    "exit_price":   exit_px,
                    "sl":           pos["sl"],
                    "tp":           pos["tp"],
                    "contracts":    pos["lots"],
                    "pnl":          raw_pnl - config.commission_rt,
                    "duration_bars":i - pos["entry_bar"],
                    "exit_reason":  "SL" if hit_sl else "TP" if hit_tp else "MEAN",
                })
                del open_positions[sym]

        # ── Check for new entry signal ───────────────────────────────────
        if sym not in open_positions and i > 0:
            prev = df.iloc[i - 1]
            if prev["signal"] != 0 and not engine.state.halted:
                tick_val = CONTRACT_SPECS.get(sym, 50.0)
                allowed, reason, lots = engine.check_entry(
                    sym, row["open"], prev["sl_price"],
                    int(prev["signal"]), tick_val, ts.to_pydatetime()
                )
                if allowed:
                    open_positions[sym] = {
                        "direction":  int(prev["signal"]),
                        "entry":      row["open"],
                        "sl":         prev["sl_price"],
                        "tp":         prev["tp_price"],
                        "lots":       lots,
                        "tick_val":   tick_val,
                        "entry_time": ts,
                        "entry_bar":  i,
                    }

    trades_df = pd.DataFrame(records)
    if trades_df.empty:
        return trades_df, pd.DataFrame(), engine.status()

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
    trades_df["equity"]         = config.account_size + trades_df["cumulative_pnl"]

    equity_df = (
        trades_df.set_index("exit_time")["equity"]
        .resample("1D").last()
        .ffill()
    )

    return trades_df, equity_df, engine.status()


# ──────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    engine = RiskEngine()
    print("Initial status:", engine.status())

    # Simulate 6 trades
    scenarios = [175, -175, 300, -175, -175, -175]
    for pnl in scenarios:
        allowed, reason, lots = engine.check_entry("ES", 4500, 4490, 1)
        print(f"  Entry allowed={allowed:5} | reason={reason:40} | lots={lots}")
        if allowed:
            engine.record_trade(pnl, "ES")

    print("\nFinal status:")
    for k, v in engine.status().items():
        print(f"  {k:<30} {v}")
