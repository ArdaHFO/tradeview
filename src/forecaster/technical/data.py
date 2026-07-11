"""OHLCV bars via yfinance (free, no API key)."""
from __future__ import annotations

import logging
from datetime import timezone

import yfinance as yf

from ..config import Config
from ..models import Bar

log = logging.getLogger(__name__)

_TIMEFRAME_SPECS: dict[str, tuple[str, str]] = {
    "30m": ("30m", "60d"),
    "1h": ("60m", "730d"),
    "1d": ("1d", "6mo"),
    "1wk": ("1wk", "5y"),
    "1mo": ("1mo", "10y"),
}


def _resolve_timeframe(cfg: Config, timeframe: str) -> tuple[str, str]:
    interval, default_period = _TIMEFRAME_SPECS.get(timeframe, (timeframe, cfg.technical_lookback_period))
    if timeframe in {"30m", "1h"}:
        return interval, cfg.intraday_lookback_period
    return interval, default_period


def fetch_bars(symbol: str, cfg: Config, timeframe: str = "1d") -> list[Bar]:
    interval, period = _resolve_timeframe(cfg, timeframe)
    df = yf.download(symbol, period=period, interval=interval,
                      progress=False, auto_adjust=False)
    if df is None or df.empty:
        log.warning("no bars for %s at %s", symbol, timeframe)
        return []
    if isinstance(df.columns, type(df.columns)) and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        ts_utc = ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)
        bars.append(Bar(
            ts=ts_utc.to_pydatetime(),
            open=float(row["Open"]), high=float(row["High"]),
            low=float(row["Low"]), close=float(row["Close"]),
            volume=float(row["Volume"] or 0.0),
        ))
    return bars


def fetch_daily_bars(symbol: str, cfg: Config) -> list[Bar]:
    return fetch_bars(symbol, cfg, "1d")


def latest_close(symbol: str, cfg: Config, timeframe: str = "1d") -> float | None:
    bars = fetch_bars(symbol, cfg, timeframe)
    return bars[-1].close if bars else None
