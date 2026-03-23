"""
MAVER Live Data Connectors
===========================
Two plug-in adapters that supply OHLCV data to the strategy engine:

1. YFinanceAdapter   — free, no account needed, rate-limited (~2 s / request)
2. IBKRAdapter       — Interactive Brokers TWS / IB Gateway (paper or live)

Both adapters expose the same interface:
    .get_historical(symbol, period, interval) → pd.DataFrame (OHLCV)
    .stream_bars(symbol, callback)            → live bar subscription

Futures ticker mapping
----------------------
Yahoo Finance uses continuous-contract tickers for futures:
    ES  → ES=F    NQ  → NQ=F    CL  → CL=F
    GC  → GC=F    YM  → YM=F    RTY → RTY=F

Interactive Brokers uses native Future contracts with explicit expiry.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Yahoo Finance continuous-contract suffix map
YF_TICKER_MAP: Dict[str, str] = {
    "ES":  "ES=F",
    "NQ":  "NQ=F",
    "CL":  "CL=F",
    "GC":  "GC=F",
    "YM":  "YM=F",
    "RTY": "RTY=F",
}

# IBKR contract specification defaults
IBKR_CONTRACT_DEFAULTS: Dict[str, Dict] = {
    "ES":  {"exchange": "CME",   "currency": "USD", "secType": "FUT"},
    "NQ":  {"exchange": "CME",   "currency": "USD", "secType": "FUT"},
    "CL":  {"exchange": "NYMEX", "currency": "USD", "secType": "FUT"},
    "GC":  {"exchange": "COMEX", "currency": "USD", "secType": "FUT"},
    "YM":  {"exchange": "CBOT",  "currency": "USD", "secType": "FUT"},
    "RTY": {"exchange": "CME",   "currency": "USD", "secType": "FUT"},
}


# ──────────────────────────────────────────────
#  Base adapter interface
# ──────────────────────────────────────────────

class DataAdapter(ABC):
    """Abstract base for all data connectors."""

    @abstractmethod
    def get_historical(
        self,
        symbol:   str,
        period:   str = "60d",
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex."""

    @abstractmethod
    def stream_bars(
        self,
        symbol:   str,
        callback: Callable[[str, pd.Series], None],
        interval: str = "5m",
    ) -> None:
        """Subscribe to live bar stream. Calls callback(symbol, bar_series) on each new bar."""

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns are lowercase and index is DatetimeIndex (UTC)."""
        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df[["open", "high", "low", "close", "volume"]].dropna()


# ──────────────────────────────────────────────
#  1. Yahoo Finance adapter
# ──────────────────────────────────────────────

class YFinanceAdapter(DataAdapter):
    """
    Historical data via yfinance (free, no authentication required).

    Install : pip install yfinance

    Limitations
    -----------
    * 5-minute bars limited to last 60 days (Yahoo policy)
    * Rate-limited: add ~2 s sleep between requests for multiple symbols
    * Yahoo futures data may have gaps near roll dates
    """

    def __init__(self, rate_limit_seconds: float = 2.0):
        self._rate_limit = rate_limit_seconds
        self._last_call  = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.monotonic()

    def get_historical(
        self,
        symbol:   str,
        period:   str = "60d",
        interval: str = "5m",
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("pip install yfinance")

        self._throttle()
        ticker = YF_TICKER_MAP.get(symbol.upper(), symbol)
        logger.info("YFinance: downloading %s (%s)", ticker, interval)
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")
        return self.normalise(raw)

    def get_all_symbols(
        self,
        symbols:  list,
        period:   str = "60d",
        interval: str = "5m",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols with rate limiting."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_historical(sym, period, interval)
                logger.info("  ✓ %s: %d bars", sym, len(result[sym]))
            except Exception as exc:
                logger.warning("  ✗ %s: %s", sym, exc)
        return result

    def stream_bars(
        self,
        symbol:   str,
        callback: Callable[[str, pd.Series], None],
        interval: str = "5m",
    ) -> None:
        """
        Polling-based pseudo-stream (yfinance has no true WebSocket feed).
        Calls callback on each newly closed bar detected.
        Runs in a background daemon thread.

        Note: For production, use Interactive Brokers for real-time data.
        """
        def _poll():
            last_bar_time: Optional[pd.Timestamp] = None
            while True:
                try:
                    df = self.get_historical(symbol, period="2d", interval=interval)
                    if not df.empty:
                        latest_time = df.index[-2]   # penultimate = last closed bar
                        if last_bar_time is None or latest_time > last_bar_time:
                            last_bar_time = latest_time
                            callback(symbol, df.iloc[-2])
                except Exception as exc:
                    logger.error("YFinance stream error for %s: %s", symbol, exc)

                bar_seconds = {"1m": 60, "5m": 300, "15m": 900}.get(interval, 300)
                time.sleep(bar_seconds)

        t = threading.Thread(target=_poll, daemon=True, name=f"yf-stream-{symbol}")
        t.start()
        logger.info("YFinance polling stream started for %s (%s)", symbol, interval)


# ──────────────────────────────────────────────
#  2. Interactive Brokers adapter
# ──────────────────────────────────────────────

class IBKRAdapter(DataAdapter):
    """
    Historical + live data via Interactive Brokers TWS API (ib_insync).

    Install     : pip install ib_insync
    Prerequisites:
      • TWS or IB Gateway running on localhost:7497 (paper) / 7496 (live)
      • API access enabled in TWS: Edit → Global Configuration → API → Settings
      • Market data subscription for futures (CME, NYMEX, COMEX, CBOT)

    Example usage
    -------------
        adapter = IBKRAdapter(port=7497)   # 7497 = paper, 7496 = live
        adapter.connect()
        df = adapter.get_historical("ES", duration="30 D", bar_size="5 mins")
        adapter.disconnect()
    """

    def __init__(
        self,
        host:     str = "127.0.0.1",
        port:     int = 7497,           # 7497 paper | 7496 live
        client_id:int = 1,
    ):
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self._ib       = None           # ib_insync.IB instance

    # ── Connection management ────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError("pip install ib_insync")

        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info(
            "IBKR connected → %s:%d (client %d) | paper=%s",
            self.host, self.port, self.client_id, self.port == 7497
        )

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IBKR disconnected")

    def _require_connection(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("Call IBKRAdapter.connect() first")

    # ── Contract builder ─────────────────────────────────────────────────────

    def _build_contract(self, symbol: str, expiry: str = "") -> "Future":
        try:
            from ib_insync import Future
        except ImportError:
            raise ImportError("pip install ib_insync")

        spec = IBKR_CONTRACT_DEFAULTS.get(symbol.upper(), {})
        return Future(
            symbol      = symbol.upper(),
            exchange    = spec.get("exchange", "CME"),
            currency    = spec.get("currency", "USD"),
            lastTradeDateOrContractMonth = expiry,   # e.g. "202506"
        )

    # ── Historical data ──────────────────────────────────────────────────────

    def get_historical(
        self,
        symbol:    str,
        period:    str = "60d",        # mapped to IBKR duration string
        interval:  str = "5m",         # mapped to IBKR bar size
        expiry:    str = "",
        what_to_show: str = "TRADES",
    ) -> pd.DataFrame:
        self._require_connection()

        # Map common period strings → IBKR duration format
        duration_map = {
            "7d": "7 D", "14d": "14 D", "30d": "30 D",
            "60d": "60 D", "90d": "90 D", "6mo": "6 M", "1y": "1 Y",
        }
        bar_size_map = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
            "30m": "30 mins", "1h": "1 hour", "1d": "1 day",
        }

        duration  = duration_map.get(period, "60 D")
        bar_size  = bar_size_map.get(interval, "5 mins")
        contract  = self._build_contract(symbol, expiry)

        logger.info("IBKR: requesting %s %s bars for %s", interval, duration, symbol)
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime   = "",
            durationStr   = duration,
            barSizeSetting= bar_size,
            whatToShow    = what_to_show,
            useRTH        = False,      # include extended hours (important for futures)
            formatDate    = 1,
        )

        if not bars:
            raise ValueError(f"No IBKR data for {symbol}")

        df = self._ib.util.df(bars)
        df = df.rename(columns={
            "date":   "datetime",
            "open":   "open",
            "high":   "high",
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        })
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
        return self.normalise(df)

    def get_all_symbols(
        self,
        symbols:  list,
        period:   str = "60d",
        interval: str = "5m",
        expiries: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        result   = {}
        expiries = expiries or {}
        for sym in symbols:
            try:
                result[sym] = self.get_historical(
                    sym, period, interval, expiry=expiries.get(sym, "")
                )
                logger.info("  ✓ %s: %d bars", sym, len(result[sym]))
                time.sleep(0.5)   # IBKR pacing rule: 50 req/10 s
            except Exception as exc:
                logger.warning("  ✗ %s: %s", sym, exc)
        return result

    # ── Live streaming ───────────────────────────────────────────────────────

    def stream_bars(
        self,
        symbol:   str,
        callback: Callable[[str, pd.Series], None],
        interval: str = "5m",
        expiry:   str = "",
    ) -> None:
        """
        Subscribe to real-time bar updates via reqRealTimeBars (5-second bars
        internally aggregated) or reqHistoricalData with keepUpToDate=True.

        Uses keepUpToDate=True approach for arbitrary bar sizes.
        """
        self._require_connection()
        bar_size_map = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
        }
        bar_size = bar_size_map.get(interval, "5 mins")
        contract = self._build_contract(symbol, expiry)

        def _on_bar_update(bars, has_new_bar):
            if has_new_bar and len(bars) >= 2:
                last_closed = bars[-2]
                series = pd.Series({
                    "open":   last_closed.open,
                    "high":   last_closed.high,
                    "low":    last_closed.low,
                    "close":  last_closed.close,
                    "volume": last_closed.volume,
                }, name=pd.Timestamp(last_closed.date))
                callback(symbol, series)

        self._ib.reqHistoricalData(
            contract,
            endDateTime    = "",
            durationStr    = "1 D",
            barSizeSetting = bar_size,
            whatToShow     = "TRADES",
            useRTH         = False,
            keepUpToDate   = True,
            formatDate     = 1,
        )
        logger.info("IBKR live bar stream started for %s (%s)", symbol, interval)


# ──────────────────────────────────────────────
#  Convenience factory
# ──────────────────────────────────────────────

def create_adapter(source: str = "yfinance", **kwargs) -> DataAdapter:
    """
    Factory function.

    Parameters
    ----------
    source : "yfinance" | "ibkr"
    kwargs : passed to the adapter constructor (e.g. port=7497 for IBKR)
    """
    if source.lower() in ("yfinance", "yf"):
        return YFinanceAdapter(**kwargs)
    elif source.lower() in ("ibkr", "ib", "interactivebrokers"):
        return IBKRAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {source!r}. Use 'yfinance' or 'ibkr'.")


# ──────────────────────────────────────────────
#  Live trading runner
# ──────────────────────────────────────────────

def run_live(
    symbols:  list,
    adapter:  DataAdapter,
    interval: str = "5m",
) -> None:
    """
    Wire the live data stream to the MAVER strategy + risk engine.

    Each new closed bar triggers signal evaluation; if a signal fires and
    the risk engine permits, the trade details are logged (order routing
    to a broker must be added separately for live execution).

    Parameters
    ----------
    symbols : list of instrument codes, e.g. ["ES", "NQ"]
    adapter : a connected DataAdapter instance
    interval: bar timeframe string
    """
    from maver_strategy import compute_indicators, generate_signals, StrategyParams, CONTRACT_SPECS
    from risk_engine import RiskEngine, RiskConfig

    params = StrategyParams()
    engine = RiskEngine(RiskConfig())

    # Prime each symbol with historical bars so indicators are warm
    print("Priming historical data…")
    hist_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = adapter.get_historical(sym, period="5d", interval=interval)
            df = compute_indicators(df, params)
            df = generate_signals(df, params)
            hist_data[sym] = df
            print(f"  ✓ {sym}: {len(df)} bars loaded")
        except Exception as exc:
            print(f"  ✗ {sym}: {exc}")

    def on_new_bar(symbol: str, bar: pd.Series) -> None:
        """Called by the adapter on each new closed bar."""
        if symbol not in hist_data:
            return

        # Append bar and recompute indicators (rolling window, efficient)
        new_row = pd.DataFrame([bar], index=[bar.name])
        hist_data[symbol] = pd.concat([hist_data[symbol], new_row]).iloc[-500:]
        df = compute_indicators(hist_data[symbol], params)
        df = generate_signals(df, params)
        hist_data[symbol] = df

        last = df.iloc[-1]
        if last["signal"] != 0:
            direction = int(last["signal"])
            side      = "LONG" if direction == 1 else "SHORT"
            tick_val  = CONTRACT_SPECS.get(symbol, 50.0)
            allowed, reason, lots = engine.check_entry(
                symbol, last["close"], last["sl_price"], direction, tick_val
            )
            if allowed:
                print(
                    f"[{bar.name}] SIGNAL {side} {symbol} | "
                    f"Entry≈{last['close']:.2f} | SL={last['sl_price']:.2f} | "
                    f"TP={last['tp_price']:.2f} | Lots={lots}"
                )
                # >>> INSERT ORDER ROUTING HERE <<<
                # e.g. adapter._ib.placeOrder(contract, LimitOrder(side, lots, last["close"]))
            else:
                print(f"[{bar.name}] Signal {side} {symbol} BLOCKED: {reason}")

    # Start streaming
    print("\nStarting live bar streams…")
    for sym in symbols:
        adapter.stream_bars(sym, on_new_bar, interval)

    print("Live trading active. Press Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ──────────────────────────────────────────────
#  CLI quick-start demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    SYMBOLS = ["ES", "NQ", "CL", "GC", "YM", "RTY"]

    print("=" * 50)
    print("  MAVER — YFinance historical data fetch demo")
    print("=" * 50)

    adapter = create_adapter("yfinance")
    data    = adapter.get_all_symbols(SYMBOLS, period="5d", interval="5m")

    for sym, df in data.items():
        print(f"  {sym}: {len(df)} bars | last close: {df['close'].iloc[-1]:.2f}")

    print("\nData ready. Pass `data` dict to run_backtest() in maver_strategy.py")
    print("Example:\n  from maver_strategy import run_backtest, performance_summary, print_report")
    print("  trades_df, equity_df = run_backtest(data)")
    print("  print_report(performance_summary(trades_df, equity_df))")
