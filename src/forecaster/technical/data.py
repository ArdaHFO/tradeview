"""Daily OHLCV bars via yfinance (free, no API key)."""
from __future__ import annotations

import logging
from datetime import timezone

import yfinance as yf

from ..config import Config
from ..models import Bar

log = logging.getLogger(__name__)


def fetch_daily_bars(symbol: str, cfg: Config) -> list[Bar]:
    df = yf.download(symbol, period=cfg.technical_lookback_period, interval="1d",
                      progress=False, auto_adjust=False)
    if df is None or df.empty:
        log.warning("no daily bars for %s", symbol)
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


def latest_close(symbol: str, cfg: Config) -> float | None:
    bars = fetch_daily_bars(symbol, cfg)
    return bars[-1].close if bars else None
