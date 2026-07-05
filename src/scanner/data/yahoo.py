"""Yahoo Finance source (free, no API key): today's movers + intraday 1m bars.

Used for the keyless 'today' replay and 'live-yf' polling modes. Yahoo 1m data
lags ~1-2 minutes and has no tick/quote stream, so order flow is approximated
from bar shape (see engine.feed_synthetic_flow). Good enough to trade the
*loop*; a paid tick feed upgrades the CVD quality later.
"""
from __future__ import annotations

import logging
from datetime import timezone

import yfinance as yf

from ..models import Bar, SymbolSnapshot

log = logging.getLogger(__name__)

SCREENS = ("day_gainers", "day_losers")     # both directions: longs and shorts


def today_movers(count: int = 100) -> list[SymbolSnapshot]:
    """Today's biggest movers from Yahoo's predefined screeners."""
    snapshots: list[SymbolSnapshot] = []
    seen: set[str] = set()
    for name in SCREENS:
        try:
            res = yf.screen(name, count=count)
        except Exception as exc:            # yahoo backend hiccups: keep going
            log.warning("yahoo screen %s failed: %s", name, exc)
            continue
        for q in res.get("quotes", []):
            sym = q.get("symbol", "")
            if not sym or sym in seen or "." in sym or "-" in sym:
                continue                     # skip dual-class/foreign suffixes
            price = q.get("regularMarketPrice")
            prev = q.get("regularMarketPreviousClose")
            vol = q.get("regularMarketVolume")
            avg = q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day")
            hi = q.get("regularMarketDayHigh") or price
            lo = q.get("regularMarketDayLow") or price
            if not price or not prev or not vol or not avg:
                continue
            seen.add(sym)
            snapshots.append(SymbolSnapshot(
                symbol=sym,
                price=float(price),
                prev_close=float(prev),
                day_volume=float(vol),
                avg_daily_volume=float(avg),
                day_high=float(hi),
                day_low=float(lo),
                float_shares=(float(q["floatShares"])
                              if q.get("floatShares") else None),
            ))
    log.info("yahoo movers: %d candidates", len(snapshots))
    return snapshots


def fetch_1m_bars(symbols: list[str]) -> dict[str, list[Bar]]:
    """Today's 1-minute bars for the given symbols, RTH only, UTC timestamps."""
    if not symbols:
        return {}
    df = yf.download(symbols, interval="1m", period="1d", group_by="ticker",
                     progress=False, threads=True, auto_adjust=False,
                     prepost=False)
    out: dict[str, list[Bar]] = {}
    if df is None or df.empty:
        return out
    single = len(symbols) == 1
    for sym in symbols:
        try:
            sub = df if single else df[sym]
        except KeyError:
            continue
        sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
        bars: list[Bar] = []
        for ts, row in sub.iterrows():
            ts_utc = ts.tz_convert(timezone.utc).to_pydatetime()
            bars.append(Bar(ts=ts_utc, open=float(row["Open"]),
                            high=float(row["High"]), low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=float(row["Volume"] or 0.0)))
        if bars:
            out[sym] = bars
    return out
